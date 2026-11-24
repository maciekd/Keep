"""PromptTab: `c` asks before wiping today's dialog history and `r` before wiping its last
30 minutes, `y` wipes through MessageHistory, anything else cancels; the outcome replaces the
key hints until the next key."""
from unittest.mock import Mock

import pytest

from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs.PromptTab import PromptTab, CONFIRM_HINT, CONFIRM_RECENT_HINT


@pytest.fixture
def prompt_manager():
    pm = Mock()
    pm.prompt = "# Prompt\nhello"
    pm.message_history.last_update_time = 1.0
    pm.message_history.render_for_display.return_value = "**User**: hi"
    pm.message_history.wipe_today.return_value = 4
    pm.message_history.wipe_recent.return_value = 2
    return pm


@pytest.fixture
def tab(prompt_manager):
    return PromptTab(prompt_manager)


def bottom_bar(tab):
    screen = TUIScreen(80, 30)
    tab.draw(screen, 0)
    return screen.row_text(1, screen.height - 1, 80).strip()


class TestClear:
    def test_c_asks_first_and_y_wipes(self, tab, prompt_manager):
        assert tab.handle_input('c') is True
        assert bottom_bar(tab) == CONFIRM_HINT.strip()
        prompt_manager.message_history.wipe_today.assert_not_called()

        assert tab.handle_input('y') is True
        prompt_manager.message_history.wipe_today.assert_called_once_with()
        assert bottom_bar(tab) == "Wiped 4 history entries for today"

    def test_any_other_key_cancels(self, tab, prompt_manager):
        tab.handle_input('c')
        assert tab.handle_input('n') is True
        prompt_manager.message_history.wipe_today.assert_not_called()
        assert bottom_bar(tab) == "Clear cancelled"

    def test_the_next_key_restores_the_hints(self, tab):
        tab.handle_input('c')
        tab.handle_input('y')
        tab.handle_input('down')
        assert "c clear" in bottom_bar(tab)
        assert "Wiped" not in bottom_bar(tab)

    def test_leaving_the_tab_disarms_the_question(self, tab, prompt_manager):
        tab.handle_input('c')
        tab.on_deactivate()
        assert tab.handle_input('y') is False   # plain scroll handling again, nothing wiped
        prompt_manager.message_history.wipe_today.assert_not_called()

    def test_wipe_scrolls_back_to_the_end(self, tab):
        tab.scroll_offset = 7
        tab.handle_input('c')
        tab.handle_input('y')
        assert tab.scroll_offset == 0

    def test_failure_is_reported_not_raised(self, tab, prompt_manager):
        prompt_manager.message_history.wipe_today.side_effect = RuntimeError("db locked")
        tab.handle_input('c')
        tab.handle_input('y')
        assert bottom_bar(tab) == "Clear failed: db locked"

    def test_r_asks_first_and_y_wipes_the_last_half_hour_only(self, tab, prompt_manager):
        assert tab.handle_input('r') is True
        assert bottom_bar(tab) == CONFIRM_RECENT_HINT.strip()
        prompt_manager.message_history.wipe_recent.assert_not_called()

        assert tab.handle_input('y') is True
        prompt_manager.message_history.wipe_recent.assert_called_once_with(30)
        prompt_manager.message_history.wipe_today.assert_not_called()
        assert "Wiped 2 history entries" in bottom_bar(tab)
        assert "30 minutes" in bottom_bar(tab)

    def test_the_half_hour_question_is_cancelled_like_the_days(self, tab, prompt_manager):
        tab.handle_input('r')
        assert tab.handle_input('c') is True   # an answer to the question, not a second one
        prompt_manager.message_history.wipe_recent.assert_not_called()
        prompt_manager.message_history.wipe_today.assert_not_called()
        assert bottom_bar(tab) == "Clear cancelled"

    def test_both_clears_are_in_the_hints_of_an_80_column_terminal(self, tab):
        assert "c clear" in bottom_bar(tab)
        assert "r clear 30m" in bottom_bar(tab)

    def test_v_shows_the_other_channels_window(self, tab, prompt_manager):
        """The text window is shown until `v` flips to voice's, and the history is rendered
        for the window shown; the hint names the other one."""
        bottom = bottom_bar(tab)
        prompt_manager.message_history.render_for_display.assert_called_with("text")
        assert "v show voice window" in bottom
        assert tab.handle_input('v') is True
        bottom = bottom_bar(tab)
        prompt_manager.message_history.render_for_display.assert_called_with("voice")
        assert "v show text window" in bottom
        assert "the voice window" in "".join(detail for _, detail in tab.bands.values())

    def test_scroll_keys_still_work(self, tab, prompt_manager):
        prompt_manager.prompt = "\n".join(f"line {i}" for i in range(100))
        bottom_bar(tab)
        assert tab.handle_input('up') is True
        assert tab.scroll_offset == 1
        assert tab.handle_input('w') is False   # the document always wraps


class TestDocument:
    def test_the_history_opens_under_a_band_naming_its_window(self, tab):
        screen = TUIScreen(80, 30)
        tab.draw(screen, 0)
        rows = [screen.row_text(0, y, 80) for y in range(screen.height)]
        band = next(y for y, row in enumerate(rows) if "MESSAGES" in row)
        assert "the text window" in rows[band]
        assert any("Prompt" in row for row in rows[:band])
        assert any("User: hi" in row for row in rows[band:])

    def test_markers_are_not_drawn(self, tab):
        screen = TUIScreen(80, 30)
        tab.draw(screen, 0)
        text = "".join(screen.row_text(0, y, 80) for y in range(screen.height))
        assert "# Prompt" not in text and "**" not in text
