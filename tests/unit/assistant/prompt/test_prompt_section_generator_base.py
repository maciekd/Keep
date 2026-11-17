import threading
import time
from unittest.mock import Mock, patch

from assistant.prompt.PromptSectionGenerator import PromptSectionGenerator


class ConcreteSection(PromptSectionGenerator):
    def __init__(self, prompt_manager=None, fetch_result="test content", fetch_error=None):
        super().__init__(prompt_manager, "TestSection")
        self._fetch_result = fetch_result
        self._fetch_error = fetch_error
        self._fetch_count = 0

    def _start_monitoring(self):
        pass

    def _fetch_content(self) -> str:
        self._fetch_count += 1
        if self._fetch_error:
            raise self._fetch_error
        return self._fetch_result


class TestPromptSectionGeneratorBase:
    @patch('assistant.prompt.PromptSectionGenerator.time.sleep')
    def test_update_content_sets_content_and_timestamp(self, mock_sleep):
        section = ConcreteSection(fetch_result="new content")
        section._update_content()
        assert section.get_content() == "new content"
        assert section.last_update_time > 0

    @patch('assistant.prompt.PromptSectionGenerator.time.sleep')
    def test_update_content_notifies_prompt_manager(self, mock_sleep):
        pm = Mock()
        section = ConcreteSection(prompt_manager=pm, fetch_result="content")
        section._update_content()
        pm.on_section_updated.assert_called_once_with("TestSection")

    @patch('assistant.prompt.PromptSectionGenerator.time.sleep')
    def test_update_content_retries_on_failure(self, mock_sleep):
        call_count = [0]
        original_fetch = None

        section = ConcreteSection(fetch_result="ok")

        def flaky_fetch():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("transient error")
            return "ok"

        section._fetch_content = flaky_fetch
        section._update_content()
        assert section.get_content() == "ok"
        assert call_count[0] == 3

    @patch('assistant.prompt.PromptSectionGenerator.time.sleep')
    def test_update_content_sets_error_after_max_retries(self, mock_sleep):
        section = ConcreteSection(fetch_error=RuntimeError("persistent"))
        section._update_content()
        content = section.get_content()
        assert "ERROR" in content
        assert section._fetch_count == 3

    @patch('assistant.prompt.PromptSectionGenerator.time.sleep')
    def test_failure_keeps_last_good_content_with_stale_warning(self, mock_sleep):
        section = ConcreteSection(fetch_result="good content")
        section._update_content()
        section._fetch_error = RuntimeError("Tessie down")
        section._update_content()
        content = section.get_content()
        assert content.startswith("good content")
        assert "[WARNING: refresh failed at" in content
        assert "may be stale" in content
        assert "ERROR" not in content

    @patch('assistant.prompt.PromptSectionGenerator.time.sleep')
    def test_repeated_failures_do_not_stack_warnings(self, mock_sleep):
        pm = Mock()
        section = ConcreteSection(prompt_manager=pm, fetch_result="good content")
        section._update_content()
        section._fetch_error = RuntimeError("Tessie down")
        section._update_content()
        content_after_first_failure = section.get_content()
        notify_count = pm.on_section_updated.call_count
        section._update_content()
        assert section.get_content() == content_after_first_failure
        assert section.get_content().count("[WARNING:") == 1
        assert pm.on_section_updated.call_count == notify_count

    @patch('assistant.prompt.PromptSectionGenerator.time.sleep')
    def test_an_outage_is_an_error_when_it_begins_and_not_again_at_every_refresh_it_lasts(self, mock_sleep):
        section = ConcreteSection(fetch_result="good content")
        section._update_content()
        section._fetch_error = RuntimeError("Tessie down")
        with patch('assistant.prompt.PromptSectionGenerator.error') as logged:
            for _ in range(4):
                section._update_content()
        assert logged.call_count == 1

    @patch('assistant.prompt.PromptSectionGenerator.time.sleep')
    def test_recovery_after_failure_clears_stale_warning(self, mock_sleep):
        section = ConcreteSection(fetch_result="good content")
        section._update_content()
        section._fetch_error = RuntimeError("Tessie down")
        section._update_content()
        section._fetch_error = None
        section._fetch_result = "fresh content"
        section._update_content()
        assert section.get_content() == "fresh content"
        assert section._refresh_failed_at is None

    @patch('assistant.prompt.PromptSectionGenerator.time.sleep')
    def test_failure_with_only_error_content_sets_new_error(self, mock_sleep):
        section = ConcreteSection(fetch_error=RuntimeError("first outage"))
        section._update_content()
        section._fetch_error = RuntimeError("second outage")
        section._update_content()
        content = section.get_content()
        assert content.startswith("ERROR")
        assert "second outage" in content

class SlowPollSection(ConcreteSection):
    """Uses the base class's polling loop, with an interval no test can wait out."""
    UPDATE_INTERVAL = 3600

    def _start_monitoring(self):
        PromptSectionGenerator._start_monitoring(self)


class TestThreadLifecycle:
    """stop() joins with a 5 s timeout, so a loop that slid back to time.sleep would leak its
    thread silently — nothing raises, the section just outlives Assistant.stop()."""

    @patch('assistant.prompt.PromptSectionGenerator.time.sleep')
    def test_stop_ends_a_thread_waiting_out_the_update_interval(self, mock_sleep):
        section = SlowPollSection(fetch_result="x")
        section.start()
        deadline = time.time() + 2
        while section._fetch_count == 0 and time.time() < deadline:
            time.sleep(0.01)
        assert section.get_content() == "x"
        started = time.time()
        section.stop()
        assert not section._thread.is_alive(), "stop must end the loop mid-wait, not after the interval"
        assert time.time() - started < 1
        assert section._fetch_count == 1

    def test_monitoring_is_restarted_after_a_failure(self):
        section = ConcreteSection()
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("CalDAV down")
            section._stop_event.set()

        section._start_monitoring = flaky
        with patch.object(section, "_wait", return_value=False) as wait:
            section._start_monitoring_wrapper()
        assert len(attempts) == 2
        wait.assert_called_once_with(5)

    def test_stop_during_the_retry_backoff_does_not_restart_monitoring(self):
        section = ConcreteSection()
        section._start_monitoring = Mock(side_effect=RuntimeError("boom"))

        def stop_during_wait(seconds):
            section._stop_event.set()
            return True

        section._wait = stop_during_wait
        section._start_monitoring_wrapper()
        assert section._start_monitoring.call_count == 1

    def test_stop_ends_a_thread_sitting_in_the_retry_backoff(self):
        section = ConcreteSection()
        failed = threading.Event()

        def fail():
            failed.set()
            raise RuntimeError("boom")

        section._start_monitoring = fail
        section.start()
        assert failed.wait(2)
        started = time.time()
        section.stop()
        assert not section._thread.is_alive()
        assert time.time() - started < 1
