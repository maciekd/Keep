"""floxy_ai_server.build() with the voice interface disabled must import nothing voice-
related — the Telegram-only install boots on the base requirements alone (Phase 6)."""
import sys
from unittest.mock import Mock, patch

import pytest

from common.ConfigManager import config

VOICE_MODULE_PREFIXES = ("interfaces.voice", "sounddevice", "torch", "speechbrain", "mlx_whisper", "silero_vad", "spacy")


def _voice_modules():
    return sorted(m for m in sys.modules if m.startswith(VOICE_MODULE_PREFIXES))


@pytest.fixture
def telegram_only_config():
    real_get = config.get
    overrides = {
        'interfaces.voice.enabled': False,
        'interfaces.console.enabled': False,
        'interfaces.telegram.enabled': True,
        'telegram_bot.bot_token': 'test-token',
    }
    with patch.object(config, 'get', side_effect=lambda key, default=None: overrides.get(key, real_get(key, default))):
        yield


@pytest.fixture
def quiet_heavy_startup():
    """Everything that would talk to the world: LLM pre-heat, integrations, agents, the bot,
    both history markers (they would otherwise land in the suite's database), and the first-run
    files (the memory stub and first_run.json would land in the developer's checkout). No LLM
    backend is built either — the CacheAuditor is stubbed whole and the handlers' DialogManagers
    get a fake per tier: the fast tier is Gemini by default and importing its client is ~0.3 s,
    paid by the first test on each worker that built one."""
    with patch('floxy_ai_server._register_llm_pre_heat'), \
         patch('assistant.DialogManager.create_llm', side_effect=lambda *a, **k: Mock(model="fake-model")), \
         patch('floxy_ai_server._post_startup_message'), \
         patch('assistant.MemoryFile.seed', return_value=False), \
         patch('assistant.FirstRun.begin_if_fresh'), \
         patch('assistant.FirstRun.Pipeline') as pipeline, \
         patch('common.StartupTracker.StartupTracker.record_shutdown'), \
         patch('assistant.integration.bootstrap.discover_integrations', return_value=Mock()) as monitor, \
         patch('assistant.integration.bootstrap.start_feeds'), \
         patch('assistant.agents.CacheAuditor.CacheAuditor'), \
         patch('assistant.agents.CodeQualityMaintainer.CodeQualityMaintainer.start'), \
         patch('assistant.agents.AutonomousActionManager.AutonomousActionManager.get_instance') as autonomous, \
         patch('assistant.prompt.PromptSectionGenerator.PromptSectionGenerator.start'), \
         patch('interfaces.telegram.TelegramTransport.TelegramTransport.start') as bot_start, \
         patch('common.TempMonitor.TempMonitor.check_installed'):
        yield {"monitor": monitor, "bot_start": bot_start, "autonomous": autonomous, "pipeline": pipeline}


@pytest.mark.unit
class TestTelegramOnlyBoot:
    def test_no_voice_module_is_imported(self, telegram_only_config, quiet_heavy_startup):
        # Other tests in the session may already have imported the voice stack; only
        # modules that appear *during* build() count.
        before = set(_voice_modules())
        import floxy_ai_server
        server = floxy_ai_server.build(8999, running_as_testsuite=False)
        try:
            new_voice_modules = sorted(set(_voice_modules()) - before)
            assert not new_voice_modules, f"voice-less boot imported: {new_voice_modules}"
            assert [i.name for i in server.interfaces] == ["http", "telegram"]
            assert server.http.deps.voice_enabled is False
            assert server.http.deps.text_stream is None and server.http.deps.microphone_routes is None
            hub = quiet_heavy_startup["bot_start"].call_args[0][0]
            assert hub.assistant is server.assistant
            assert server.interface("telegram").transport in hub.transports
            quiet_heavy_startup["monitor"].return_value.start.assert_called_once()
        finally:
            server.stop()

    def test_handlers_are_built_without_voice_factories(self, telegram_only_config, quiet_heavy_startup):
        import floxy_ai_server
        server = floxy_ai_server.build(8999, running_as_testsuite=False)
        try:
            handler = server.assistant._handler_factory()
            assert handler.playback is None and handler.turn_gate is None
            assert handler.voice_factory is None and handler.text_stream is None
        finally:
            server.stop()

    def test_build_installs_the_log_masker_from_the_vault(self, telegram_only_config, quiet_heavy_startup):
        import common.logging as logging_mod
        import floxy_ai_server
        old = logging_mod._masker
        logging_mod.set_log_masker(None)
        server = floxy_ai_server.build(8999, running_as_testsuite=False)
        try:
            from common.vault.Redactor import Redactor
            assert isinstance(logging_mod._masker.__self__, Redactor)
            assert logging_mod._masker.__func__ is Redactor.mask
        finally:
            server.stop()
            logging_mod._masker = old

    def test_build_installs_the_system_note_history_and_the_suite_detaches_it(self, telegram_only_config, quiet_heavy_startup):
        # The autouse fixture drops it again afterwards; without that, an integration test's
        # system_note() on the same worker would write into this history.
        from assistant import SystemNote
        from common.FloxyDatabase import FloxyDatabase
        import floxy_ai_server
        assert SystemNote.history() is None
        server = floxy_ai_server.build(8999, running_as_testsuite=False)
        try:
            assert SystemNote.history() is server.assistant.prompt_manager.message_history
            assert FloxyDatabase.get_instance().db_path != "floxy.db"
        finally:
            server.stop()

    def test_health_route_answers_without_voice(self, telegram_only_config, quiet_heavy_startup):
        import asyncio
        import floxy_ai_server
        server = floxy_ai_server.build(8999, running_as_testsuite=False)
        try:
            async def probe():
                r = await server.http.app.test_client().get("/health")
                return r.status_code, await r.get_json()
            from common import Version
            assert asyncio.run(probe()) == (200, {"status": "ok", "version": Version.detail()})
        finally:
            server.stop()


@pytest.mark.unit
class TestDisabledSonosIsLeftAlone:
    def test_build_does_not_build_the_music_integration(self, telegram_only_config, quiet_heavy_startup):
        # Only the IntegrationMonitor builds it (mocked away here, as on an install where
        # Sonos & Spotify is disabled): built at startup it discovers speakers on its own.
        from integrations.music.MusicIntegration import MusicIntegration
        MusicIntegration.reset_instance()
        import floxy_ai_server
        server = floxy_ai_server.build(8999, running_as_testsuite=False)
        try:
            assert MusicIntegration.peek_instance() is None
        finally:
            server.stop()


@pytest.mark.unit
class TestCrashTraces:
    def test_traces_go_to_the_real_stderr_not_the_logging_wrapper(self):
        # faulthandler needs a file descriptor; sys.stderr is a StderrLogger once _parse_args ran.
        import floxy_ai_server
        with patch.object(sys, 'stderr', Mock(spec=[])), patch('floxy_ai_server.faulthandler') as fh:
            floxy_ai_server._enable_crash_traces()
        assert fh.enable.call_args.kwargs["file"] is sys.__stderr__
        assert fh.enable.call_args.kwargs["all_threads"] is True

    def test_a_process_without_stderr_starts_anyway(self):
        import floxy_ai_server
        with patch.object(sys, '__stderr__', None), patch('floxy_ai_server.faulthandler') as fh:
            floxy_ai_server._enable_crash_traces()
        fh.enable.assert_not_called()


@pytest.mark.unit
class TestInterfaceFlags:
    def test_voice_is_off_unless_the_config_says_so(self, quiet_heavy_startup):
        real_get = config.get
        overrides = {'interfaces.console.enabled': False, 'interfaces.telegram.enabled': False}
        # No interfaces.voice key at all: the default decides.
        with patch.object(config, 'get', side_effect=lambda key, default=None:
                          overrides.get(key, default if key == 'interfaces.voice.enabled' else real_get(key, default))):
            import floxy_ai_server
            server = floxy_ai_server.build(8999, running_as_testsuite=False)
        try:
            assert [i.name for i in server.interfaces] == ["http"]
        finally:
            server.stop()
        from interfaces.Interface import Interface
        from interfaces.http.HttpInterface import HttpInterface
        assert Interface.enabled_by_default is True and HttpInterface.enabled_by_default is True
        from common import ConfigStore
        assert ConfigStore.INTERFACE_ENABLED_BY_DEFAULT == {"voice": False}

    def test_every_interface_flag_is_honoured(self, quiet_heavy_startup):
        real_get = config.get
        overrides = {'interfaces.voice.enabled': False, 'interfaces.console.enabled': False,
                     'interfaces.telegram.enabled': False}
        with patch.object(config, 'get', side_effect=lambda key, default=None: overrides.get(key, real_get(key, default))):
            import floxy_ai_server
            server = floxy_ai_server.build(8999, running_as_testsuite=False)
        try:
            assert [i.name for i in server.interfaces] == ["http"]
            quiet_heavy_startup["bot_start"].assert_not_called()
        finally:
            server.stop()

    def test_telegram_starts_on_the_flag_alone_the_token_comes_later(self):
        """Without a token the interface still starts: the bot is listed as not configured
        and restart_integration("Telegram") brings it up once config_request stored one."""
        from interfaces.telegram.TelegramInterface import TelegramInterface
        real_get = config.get
        overrides = {'interfaces.telegram.enabled': True, 'telegram_bot.bot_token': ''}
        with patch.object(config, 'get', side_effect=lambda key, default=None: overrides.get(key, real_get(key, default))):
            assert TelegramInterface.enabled(config) is True
        overrides['interfaces.telegram.enabled'] = False
        with patch.object(config, 'get', side_effect=lambda key, default=None: overrides.get(key, real_get(key, default))):
            assert TelegramInterface.enabled(config) is False

    def test_testsuite_mode_skips_telegram(self):
        import floxy_ai_server
        assert floxy_ai_server._enabled("telegram", running_as_testsuite=True) is False
        assert floxy_ai_server._enabled("http", running_as_testsuite=True) is True


@pytest.mark.unit
class TestStop:
    def test_stop_ends_everything_build_started(self, telegram_only_config, quiet_heavy_startup):
        import floxy_ai_server
        server = floxy_ai_server.build(8999, running_as_testsuite=False)
        assistant = server.assistant
        try:
            assistant.start()  # main() does this once it serves
            assert assistant._replenish_thread.is_alive()
        finally:
            server.stop()
        assert assistant._stop_event.is_set() and assistant._replenish_thread is None
        assert assistant.pooled_handlers() == 0
        quiet_heavy_startup["autonomous"].return_value.stop.assert_called_once()
        server.stop()  # idempotent
        quiet_heavy_startup["autonomous"].return_value.stop.assert_called_once()

    def test_a_failing_assistant_stop_still_shuts_logging_down(self, telegram_only_config, quiet_heavy_startup):
        # assistant.stop() now fans out to the prompt manager's section threads; one of them
        # raising must not cost the shutdown lines at the tail of floxy.out.
        import floxy_ai_server
        server = floxy_ai_server.build(8999, running_as_testsuite=False)
        real_assistant = server.assistant
        server.assistant = Mock(stop=Mock(side_effect=RuntimeError("prompt manager wedged")))
        try:
            with patch("common.logging.shutdown_logging") as shutdown:
                server.stop()
        finally:
            real_assistant.stop()
        server.assistant.stop.assert_called_once()
        shutdown.assert_called_once()


@pytest.mark.unit
class TestConfigCheck:
    def test_missing_required_keys_stop_the_start_with_the_fix_spelled_out(self):
        import floxy_ai_server
        with patch.object(config, "missing_required", return_value=["api_keys.anthropic", "api_keys.gemini", "vault.key"]), \
                patch.object(config, "config_path", return_value="/srv/config.yaml"), \
                patch.object(type(config), "created_skeleton", True), \
                patch("floxy_ai_server.error") as err:
            with pytest.raises(SystemExit) as exc:
                floxy_ai_server._verify_config_or_exit()
        assert exc.value.code == 1
        lines = [c.args[0] for c in err.call_args_list]
        assert any("Please set api_keys.anthropic, api_keys.gemini and vault.key in config.yaml and restart." in l for l in lines)
        assert any("was written at" in l for l in lines)

    def test_a_local_smart_tier_needs_no_cloud_key(self):
        import floxy_ai_server
        with patch("assistant.llm.parse_llm_config", return_value=("OllamaLLM", "qwen3.8:27b")), \
                patch("assistant.llm.tier_api_key_name", return_value=None):
            assert floxy_ai_server._required_keys() == ("vault.key",)

    def test_a_cloud_smart_tier_requires_its_own_key(self):
        import floxy_ai_server
        with patch("assistant.llm.parse_llm_config", return_value=("GeminiLLM", "gemini-3.5-flash")), \
                patch("assistant.llm.tier_api_key_name", return_value="api_keys.gemini"):
            assert floxy_ai_server._required_keys() == ("api_keys.gemini", "vault.key")

    def test_one_missing_key_is_named_alone(self):
        import floxy_ai_server
        with patch.object(config, "missing_required", return_value=["api_keys.gemini"]), \
                patch.object(config, "config_path", return_value="/srv/config.yaml"), \
                patch("floxy_ai_server.error") as err:
            with pytest.raises(SystemExit):
                floxy_ai_server._verify_config_or_exit()
        assert any("Please set api_keys.gemini in config.yaml and restart." in c.args[0] for c in err.call_args_list)

@pytest.mark.unit
class TestFirstConversation:
    def test_with_the_console_the_brief_waits_for_its_startup_intro(self):
        import floxy_ai_server

        class FakeConsole:
            name = "console"

            def __init__(self):
                self.waiting = []

            def after_intro(self, fn):
                self.waiting.append(fn)

        class FakeServer:
            def __init__(self, console):
                self.interfaces = [type("Http", (), {"name": "http"})(), console]

        console = FakeConsole()
        with patch("assistant.FirstRun.in_ftue", return_value=True), \
                patch("assistant.FirstRun.brief", return_value="the brief"), \
                patch("interfaces.chat.ChatHub.ChatHub.get_instance") as hub:
            floxy_ai_server._open_first_conversation(FakeServer(console))
            hub.return_value.open_conversation.assert_not_called()
            assert len(console.waiting) == 1
            console.waiting[0]()
        hub.return_value.open_conversation.assert_called_once_with("the brief")

    def test_without_the_console_the_brief_goes_out_at_once(self):
        import floxy_ai_server

        class FakeServer:
            interfaces = [type("Telegram", (), {"name": "telegram"})()]

        with patch("assistant.FirstRun.in_ftue", return_value=True), \
                patch("assistant.FirstRun.brief", return_value="the brief"), \
                patch("interfaces.chat.ChatHub.ChatHub.get_instance") as hub:
            floxy_ai_server._open_first_conversation(FakeServer())
        hub.return_value.open_conversation.assert_called_once_with("the brief")

    def test_the_startup_announcement_goes_out_quietly_as_a_row_already_in_history(self):
        """The startup row is in the history before the interfaces start, so the chats that
        draw the history must skip the announcement — only the others (Telegram) get it,
        without a notification sound: a restart may happen at night."""
        import floxy_ai_server
        from assistant.Notifier import set_notifier
        chats = Mock()
        set_notifier(chats)
        try:
            server = floxy_ai_server.Server(Mock(), [], [], None, running_as_testsuite=False)
            floxy_ai_server._announce_startup(server)
            chats.send_message.assert_not_called()
            server.startup_announcement = "Server is up again"
            floxy_ai_server._announce_startup(server)
        finally:
            set_notifier(None)
        chats.send_message.assert_called_once_with("Server is up again", disable_notification=True, in_history=True)

    def test_an_installed_system_waits_to_be_spoken_to(self):
        import floxy_ai_server
        with patch("assistant.FirstRun.in_ftue", return_value=False), \
                patch("interfaces.chat.ChatHub.ChatHub.get_instance") as hub:
            floxy_ai_server._open_first_conversation()
        hub.assert_not_called()


@pytest.mark.unit
class TestFirstRunPipeline:
    def test_build_starts_the_pipeline_with_the_history_and_stop_ends_it(self, telegram_only_config, quiet_heavy_startup):
        import floxy_ai_server
        pipeline = quiet_heavy_startup["pipeline"]
        server = floxy_ai_server.build(8999, running_as_testsuite=False)
        try:
            pipeline.assert_called_once_with(server.assistant.prompt_manager.message_history)
            pipeline.return_value.start.assert_called_once()
            assert pipeline.return_value in server.agents
        finally:
            server.stop()
        pipeline.return_value.stop.assert_called_once()

    def test_the_test_suite_boot_has_no_pipeline(self, telegram_only_config, quiet_heavy_startup):
        import floxy_ai_server
        server = floxy_ai_server.build(8999, running_as_testsuite=True)
        try:
            quiet_heavy_startup["pipeline"].assert_not_called()
        finally:
            server.stop()

    def test_the_stub_is_seeded_and_the_pipeline_armed_before_the_prompt_manager_reads_the_bank(
            self, telegram_only_config, quiet_heavy_startup):
        """begin_if_fresh must get seed()'s answer — False here would never arm a fresh install —
        and both must run before the prompt manager's memory section first reads memory.md."""
        import floxy_ai_server
        from assistant.prompt.PromptManager import PromptManager
        order = []
        original_init = PromptManager.__init__

        def recording_init(self, *args, **kwargs):
            order.append("prompt_manager")
            original_init(self, *args, **kwargs)

        with patch('assistant.MemoryFile.seed', side_effect=lambda: order.append("seed") or True), \
             patch('assistant.FirstRun.begin_if_fresh', side_effect=lambda seeded: order.append(("begin", seeded))), \
             patch.object(PromptManager, "__init__", recording_init):
            server = floxy_ai_server.build(8999, running_as_testsuite=False)
        try:
            assert order == ["seed", ("begin", True), "prompt_manager"]
        finally:
            server.stop()
