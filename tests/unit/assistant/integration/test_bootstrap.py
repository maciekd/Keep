"""bootstrap: the launcher's integration wiring. Every integration is registered by class
so one whose constructor raises (a missing config section) lands as a NOT_CONFIGURED
placeholder the assistant can list and restart, instead of being skipped at startup
(plans/SELF_SETUP_PLAN.md §3.3)."""
from unittest.mock import Mock, patch

import pytest

from assistant.integration import bootstrap
from assistant.integration.Integration import Integration
from assistant.integration.IntegrationMonitor import IntegrationMonitor
from assistant.tools.WebAccess import WebAccess


@pytest.fixture
def monitor():
    fake = Mock(spec=IntegrationMonitor)
    with patch.object(IntegrationMonitor, "get_instance", return_value=fake):
        yield fake


@pytest.mark.unit
class TestDiscoverIntegrations:
    def test_registers_every_integration_by_class_plus_web_access(self, monitor):
        from integrations.icloud.ICloudIntegration import ICloudIntegration
        from integrations.tesla.TeslaIntegration import TeslaIntegration
        from integrations.ring.RingIntegration import RingIntegration
        from integrations.imap.ImapMailIntegration import ImapMailIntegration
        from assistant.tools.WebAccess import WebAccess

        assert bootstrap.discover_integrations() is monitor

        registered = [call.args[0] for call in monitor.register_class.call_args_list]
        for cls in (ICloudIntegration, TeslaIntegration, RingIntegration, ImapMailIntegration, WebAccess):
            assert registered.count(cls) == 1, cls.__name__
        # Only concrete integrations: never the base class, and none twice even though
        # subpackages re-export what they import.
        assert Integration not in registered
        assert len(registered) == len(set(registered))
        # Nothing is registered by instance — that path skipped a raising constructor.
        monitor.register.assert_not_called()

    def test_every_other_integration_starts_its_own_background_work_through_the_hook(self, monitor):
        bootstrap.discover_integrations()
        hooks = [call.kwargs.get("start") for call in monitor.register_class.call_args_list if call.args[0] is not WebAccess]
        assert hooks and all(hooks)
        integration = Mock(name="integration")
        hooks[0](integration)
        integration.start.assert_called_once_with()

    def test_feeds_are_left_to_start_feeds(self, monitor):
        from integrations.imessage.IMessageIntegration import IMessageIntegration
        from integrations.telegram_feed.TelegramFeedIntegration import TelegramFeedIntegration
        from integrations.photos.PhotosIntegration import PhotosIntegration
        bootstrap.discover_integrations()
        registered = [call.args[0] for call in monitor.register_class.call_args_list]
        assert IMessageIntegration not in registered
        assert TelegramFeedIntegration not in registered
        assert PhotosIntegration not in registered

    def test_the_monitor_is_notified_once_after_the_last_registration(self, monitor):
        bootstrap.discover_integrations()
        calls = [name for name, _args, _kwargs in monitor.method_calls]
        assert calls[-1] == "notify" and calls.count("notify") == 1
        assert calls.index("notify") > max(i for i, name in enumerate(calls) if name == "register_class")


@pytest.mark.unit
class TestStartFeeds:
    def test_feeds_are_restartable_and_start_on_the_runner(self, monitor):
        from assistant.integration.FeedRunner import Feeds
        from integrations.imessage.IMessageIntegration import IMessageIntegration
        from integrations.telegram_feed.TelegramFeedIntegration import TelegramFeedIntegration
        from integrations.photos.PhotosIntegration import PhotosIntegration
        from integrations.indigo.IndigoIntegration import IndigoIntegration
        from integrations.home_assistant.HomeAssistantIntegration import HomeAssistantIntegration
        from integrations.whatsapp.WhatsAppIntegration import WhatsAppIntegration
        from integrations.google.GoogleDocsIntegration import GoogleDocsIntegration
        history = Mock(name="message_history")

        bootstrap.start_feeds(history)

        by_class = {call.args[0]: call.kwargs["start"] for call in monitor.register_class.call_args_list}
        assert set(by_class) == {IMessageIntegration, TelegramFeedIntegration, PhotosIntegration, GoogleDocsIntegration,
                                 IndigoIntegration, HomeAssistantIntegration, WhatsAppIntegration}
        # The hook is what register_class runs now and again on every restart_integration.
        with patch.object(Feeds, "get_instance") as feeds:
            for cls, start in by_class.items():
                feed = Mock(name=cls.__name__)
                start(feed)
                feeds.return_value.start.assert_called_with(feed, history)


@pytest.mark.unit
class TestIntegrationModules:
    def test_a_tab_module_is_left_to_the_console(self):
        walked = [(None, "integrations.tesla", True), (None, "integrations.tesla.TeslaIntegration", False),
                  (None, "integrations.tesla.TeslaTab", False)]
        with patch.object(bootstrap.pkgutil, "walk_packages", return_value=walked), \
                patch.object(bootstrap.importlib, "import_module", side_effect=lambda name: name):
            modules = list(bootstrap._integration_modules())
        assert modules == ["integrations.tesla", "integrations.tesla.TeslaIntegration"]

    def test_a_module_that_fails_to_import_is_left_out_and_its_neighbours_still_load(self):
        """A local integration folder with a syntax error or a missing package costs only
        itself: the walk goes on past it and the error names the module."""
        walked = [(None, "integrations.tesla", True), (None, "integrations.tesla.TeslaIntegration", False),
                  (None, "integrations.podcasts", True), (None, "integrations.podcasts.PodcastsIntegration", False)]

        def import_or_fail(name):
            if name == "integrations.podcasts.PodcastsIntegration":
                raise ModuleNotFoundError("No module named 'feedparser'")
            return name

        with patch.object(bootstrap.pkgutil, "walk_packages", return_value=walked), \
                patch.object(bootstrap.importlib, "import_module", side_effect=import_or_fail), \
                patch.object(bootstrap, "error") as error:
            modules = list(bootstrap._integration_modules())
        assert modules == ["integrations.tesla", "integrations.tesla.TeslaIntegration", "integrations.podcasts"]
        assert error.call_count == 1
        assert "integrations.podcasts.PodcastsIntegration" in error.call_args.args[0]
        assert "feedparser" in error.call_args.args[0]

    def test_a_package_the_walker_cannot_read_is_logged_and_the_walk_goes_on(self):
        """pkgutil reports a package whose __init__ fails through `onerror` and continues
        without it; without the hook it would raise and no integration after it would load."""
        walked = [(None, "integrations.tesla", True), (None, "integrations.tesla.TeslaIntegration", False)]

        def walk(path, prefix, onerror):
            onerror("integrations.broken")
            return iter(walked)

        with patch.object(bootstrap.pkgutil, "walk_packages", side_effect=walk), \
                patch.object(bootstrap.importlib, "import_module", side_effect=lambda name: name), \
                patch.object(bootstrap, "error") as error:
            modules = list(bootstrap._integration_modules())
        assert modules == ["integrations.tesla", "integrations.tesla.TeslaIntegration"]
        assert error.call_count == 1 and "integrations.broken" in error.call_args.args[0]
