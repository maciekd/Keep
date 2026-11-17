"""The calendar section: the coming week of every calendar account, refreshed every 5 minutes,
standing under its own heading in the prompt's data block and running only while a calendar
source is on."""
from unittest.mock import Mock, patch

import pytest

from assistant.integration.calendar.CalendarSource import CalendarSource
from assistant.prompt.sections.CalendarSectionGenerator import CalendarSectionGenerator


@pytest.mark.unit
class TestCalendarSectionGenerator:
    def test_the_content_is_the_composites_coming_week(self):
        with patch('assistant.prompt.sections.CalendarSectionGenerator.CalendarAccess.current') as current:
            current.return_value.coming_week.return_value = "Calendars shown below: Home\n\n(No upcoming events)"
            section = CalendarSectionGenerator(None)
            assert section._fetch_content() == "Calendars shown below: Home\n\n(No upcoming events)"

    def test_a_week_no_account_could_read_keeps_the_week_the_section_has(self):
        from assistant.integration.calendar.CalendarAccess import CalendarUnavailable
        section = CalendarSectionGenerator(Mock())
        with patch('assistant.prompt.sections.CalendarSectionGenerator.CalendarAccess.current') as current, \
                patch('assistant.prompt.PromptSectionGenerator.time.sleep'):
            current.return_value.coming_week.return_value = "Calendars shown below: Home\n\nDentist Charlotte"
            section._update_content()
            current.return_value.coming_week.side_effect = CalendarUnavailable("no calendar account could be read")
            section._update_content()
        assert "Dentist Charlotte" in section.get_content()
        assert current.return_value.coming_week.call_count == 4     # the read, then the three attempts

    def test_the_heading_is_a_contract_with_the_toolkits_that_point_the_llm_at_it(self):
        """The section has no placeholder of its own: it stands in the data block under its
        heading, which is the name the calendar and music toolkit references send the LLM to."""
        from assistant.prompt.PromptManager import PromptManager
        from common.Paths import RES_DIR
        template = (RES_DIR / "prompt.md").read_text(encoding="utf-8")
        heading = CalendarSectionGenerator.heading
        assert heading.startswith("# ") and PromptManager.DATA_BLOCK in template
        assert "{" + CalendarSectionGenerator.key.upper() + "}" not in template and heading not in template
        assert heading.lstrip("# ") in (RES_DIR / "toolkits" / "calendar.md").read_text(encoding="utf-8")

    def test_it_runs_only_while_the_monitor_has_a_calendar_source_on(self):
        """2026-10-07: production's first fetch ran before CalDAV was registered and the prompt
        said "not configured" for the whole 15-minute interval — now the section is not started
        until a source is registered on, and never for an install without one."""
        def monitor(verdict):
            return Mock(contract_enabled=Mock(side_effect=lambda c: verdict if c is CalendarSource else None),
                        folder_enabled=Mock(return_value=True))
        assert CalendarSectionGenerator.folder() is None
        assert CalendarSectionGenerator.enabled_by(monitor(True)) is True
        assert CalendarSectionGenerator.enabled_by(monitor(False)) is False
        assert CalendarSectionGenerator.enabled_by(monitor(None)) is False
