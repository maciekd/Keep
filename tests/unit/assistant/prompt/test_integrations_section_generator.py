"""IntegrationsSectionGenerator: the enabled integrations by state, in the prompt while setup
is in progress and absent after."""
from unittest.mock import patch

import pytest

from assistant.prompt.sections.IntegrationsSectionGenerator import IntegrationsSectionGenerator

MODULE = "assistant.prompt.sections.IntegrationsSectionGenerator.IntegrationsSectionGenerator"


def _status(name, state, last_error=None):
    return {"name": name, "state": state, "last_error": last_error, "config_keys": []}


STATUSES = [
    _status("Photos", "ok"),
    _status("Arlo", "disabled"),
    _status("CalDAV", "not_configured"),
    _status("iMessage", "ok"),
    _status("Self Improvement", "error", "Claude Code is not installed"),
    _status("Tesla", "unknown"),
    _status("Telegram Feed", "needs_2fa"),
]


@pytest.fixture
def section():
    return IntegrationsSectionGenerator(prompt_manager=None)


@pytest.fixture(autouse=True)
def no_real_probe():
    with patch(f"{MODULE}.full_disk_access", return_value=False):
        yield


@pytest.mark.usefixtures("on_a_mac")
@pytest.mark.unit
class TestRender:
    def test_groups_the_enabled_integrations_by_state_and_leaves_the_disabled_out(self):
        text = IntegrationsSectionGenerator.render(STATUSES)
        lines = text.splitlines()
        assert lines[0] == "# INTEGRATIONS"
        assert lines[1] == "On: iMessage, Photos, Tesla."
        assert lines[2] == "On but not configured: CalDAV."
        assert lines[3] == "On with a problem: Self Improvement (error: Claude Code is not installed), Telegram Feed (needs 2FA)."
        assert "Arlo" not in text

    def test_the_last_line_is_the_full_disk_access_verdict(self):
        assert IntegrationsSectionGenerator.render(STATUSES, True).splitlines()[-1] == \
            "Full Disk Access for this server process: granted."
        assert "not granted" in IntegrationsSectionGenerator.render(STATUSES, False).splitlines()[-1]
        assert "Messages" in IntegrationsSectionGenerator.render(STATUSES, None).splitlines()[-1]

    def test_the_fetched_text_carries_the_probes_verdict(self, section):
        with patch(f"{MODULE}.in_ftue", return_value=True), patch(f"{MODULE}.statuses", return_value=STATUSES), \
                patch(f"{MODULE}.full_disk_access", return_value=True):
            assert "Full Disk Access for this server process: granted." in section._fetch_content()

    def test_a_long_error_is_cut(self):
        text = IntegrationsSectionGenerator.render([_status("Gmail", "error", "x" * 500)])
        assert "x" * IntegrationsSectionGenerator.ERROR_CHARS + ")" in text
        assert "x" * (IntegrationsSectionGenerator.ERROR_CHARS + 1) not in text

    def test_the_problem_and_unconfigured_lines_appear_only_with_members(self):
        text = IntegrationsSectionGenerator.render([_status("Photos", "ok")])
        assert text.splitlines()[:2] == ["# INTEGRATIONS", "On: Photos."]
        assert len(text.splitlines()) == 3     # the grant's line is always there

    def test_nothing_enabled_says_so(self):
        assert "nothing yet" in IntegrationsSectionGenerator.render([_status("Arlo", "disabled")])

    def test_the_section_stays_small(self):
        # The point of names only: a full install's state in well under a kilobyte.
        many = [_status(f"Integration {i}", "ok") for i in range(20)]
        assert len(IntegrationsSectionGenerator.render(many).encode()) < 600


@pytest.mark.unit
class TestFetchContent:
    def test_absent_when_setup_is_not_in_progress(self, section):
        with patch(f"{MODULE}.in_ftue", return_value=False), patch(f"{MODULE}.statuses") as statuses:
            assert section._fetch_content() == ""
        statuses.assert_not_called()

    def test_present_while_setup_is_in_progress(self, section):
        with patch(f"{MODULE}.in_ftue", return_value=True), patch(f"{MODULE}.statuses", return_value=STATUSES):
            assert section._fetch_content().startswith("# INTEGRATIONS\nOn: iMessage, Photos, Tesla.")

    def test_the_predicate_is_first_runs(self, section):
        with patch("assistant.FirstRun.in_ftue", return_value=False) as predicate:
            assert section._fetch_content() == ""
        predicate.assert_called_once()


@pytest.mark.unit
class TestPollLogging:
    """The section is polled every 20 s; the log must not carry a line per poll."""

    def _poll(self, section):
        # What the base class does after _fetch_content: keeps the text for the next comparison.
        section.content = section._fetch_content()

    def test_the_summary_is_logged_once_for_an_unchanged_state(self, section):
        with patch(f"{MODULE}.in_ftue", return_value=True), patch(f"{MODULE}.statuses", return_value=STATUSES), \
                patch("assistant.prompt.sections.IntegrationsSectionGenerator.info") as info:
            for _ in range(3):
                self._poll(section)
        assert info.call_count == 2  # the verdict once, the summary once

    def test_a_changed_state_is_logged_again(self, section):
        with patch(f"{MODULE}.in_ftue", return_value=True), \
                patch(f"{MODULE}.statuses", side_effect=[STATUSES, STATUSES, STATUSES[:1]]), \
                patch("assistant.prompt.sections.IntegrationsSectionGenerator.info") as info:
            for _ in range(3):
                self._poll(section)
        assert info.call_count == 3

    def test_setup_over_is_logged_once(self, section):
        with patch(f"{MODULE}.in_ftue", return_value=False), \
                patch("assistant.prompt.sections.IntegrationsSectionGenerator.info") as info:
            for _ in range(3):
                self._poll(section)
        assert info.call_count == 1

    def test_the_poll_asks_first_run_without_its_trace_lines(self, section):
        with patch("assistant.FirstRun.in_ftue", return_value=False) as predicate:
            section._fetch_content()
        assert predicate.call_args.kwargs == {"verbose": False}


@pytest.mark.unit
class TestOnAServerThatIsNotAMac:
    @pytest.fixture(autouse=True)
    def linux(self, monkeypatch):
        from common import Platform
        monkeypatch.setattr(Platform, "current", lambda: Platform.LINUX)

    def test_the_last_line_names_the_system_and_what_needs_a_mac_instead_of_the_grant(self):
        statuses = STATUSES + [dict(_status("Reminders", "disabled"), needs_mac=True),
                               dict(_status("HomeClaw", "disabled"), needs_mac=True)]
        text = IntegrationsSectionGenerator.render(statuses, full_disk_access=False)
        last = text.splitlines()[-1]
        assert "Full Disk Access" not in text
        assert "Linux" in last and "HomeClaw, Reminders" in last and "microphones" in last
        assert "Arlo" not in last, "an integration that is merely disabled can be enabled here"

    def test_an_integration_that_needs_a_mac_is_on_no_other_line(self):
        text = IntegrationsSectionGenerator.render([dict(_status("Photos", "disabled"), needs_mac=True)])
        assert text.splitlines()[1] == "On: nothing yet." and text.count("Photos") == 1
