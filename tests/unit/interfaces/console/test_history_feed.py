"""HistoryFeed: the dialog history flows into the chat pane — backfill, live inserts by
timestamp, headers per role — and the console's own turns are adopted, not duplicated."""
import time
from unittest.mock import patch

import pytest

from interfaces.console.HistoryFeed import HistoryFeed, normalize
from interfaces.console.widgets.ChatPane import (KIND_ASSISTANT, KIND_DIFF, KIND_NOTICE, KIND_SYSTEM, KIND_USER,
                                                  ChatPane)

T0 = 1_700_000_000.0


def entry(role, content, ts, source=None, location=None):
    return {"role": role, "content": content, "timestamp": ts, "source": source, "location": location}


@pytest.fixture
def history():
    from assistant.prompt.MessageHistory import MessageHistory
    with patch.object(MessageHistory, '_load_history'):
        return MessageHistory()


@pytest.fixture
def pane():
    return ChatPane()


@pytest.fixture
def feed(history, pane):
    feed = HistoryFeed(history, pane)
    feed.start()
    return feed


def land(history, *entries):
    """Persisted rows arriving in the history (what add_exchange / the log monitor do)."""
    history.history = list(history.history) + list(entries)
    history._notify_changed()


def shown(pane):
    return [(e.kind, e.header, e.body.text) for e in pane.entries]


class TestBackfill:
    def test_every_role_lands_with_its_header(self, history, pane):
        history.history = [
            entry("system", "Server started up successfully.", T0, source="startup"),
            entry("marcus", "Where's my Model X?", T0 + 10, source="VoiceToVoice", location="Kitchen"),
            entry("assistant", "Parked at <b>Kyrkogatan</b>.", T0 + 10.001),
            entry("system", "Charlotte got home.", T0 + 20),
        ]
        HistoryFeed(history, pane).start()
        assert shown(pane) == [
            (KIND_SYSTEM, "System (startup)", "Server started up successfully."),
            (KIND_USER, "Marcus (VoiceToVoice in Kitchen)", "Where's my Model X?"),
            (KIND_ASSISTANT, "Assistant", "Parked at <b>Kyrkogatan</b>."),
            (KIND_SYSTEM, "System", "Charlotte got home."),
        ]
        assert all(not e.body.animated for e in pane.entries)

    def test_stop_unsubscribes(self, history, pane, feed):
        feed.stop()
        land(history, entry("system", "late", T0))
        assert pane.entries == []


class TestLiveInserts:
    def test_new_rows_are_added_once_in_timestamp_order(self, history, pane, feed):
        land(history, entry("system", "second", T0 + 2))
        land(history, entry("system", "second", T0 + 2), entry("system", "first", T0 + 1))
        assert [e.body.text for e in pane.entries] == ["first", "second"]

    def test_prompt_section_note_is_hidden_by_default(self, history, pane, feed):
        history.add_transient_entry("system", "Prompt section `TeslaInfo` updated:\n-Charlottes <Model 3>",
                                    T0, source="prompt_delta", section="tesla")
        assert pane.entries == []

    def test_prompt_section_note_shows_as_a_diff_entry_when_enabled(self, history, pane, feed):
        with patch('interfaces.console.HistoryFeed.SHOW_PROMPT_DELTAS', True):
            history.add_transient_entry("system", "Prompt section `TeslaInfo` updated:\n-Charlottes <Model 3>",
                                        T0, source="prompt_delta", section="tesla")
            assert shown(pane)[0] == (KIND_DIFF, "System (prompt_delta)",
                                      "Prompt section `TeslaInfo` updated:\n-Charlottes <Model 3>")
            # Superseding the note keeps the chat a log of what happened
            history.clear_transient_entries_for_section("tesla")
            assert len(pane.entries) == 1

    def test_messaging_rows_are_marked_up_for_the_pane(self, history, pane, feed):
        land(history, entry("system", "iMessage received:\nFrom Charlotte to Marcus: hej", T0, source="iMessage"))
        assert shown(pane) == [(KIND_SYSTEM, "System (iMessage)", "<b>Charlotte → Marcus</b> hej")]

class TestAdoption:
    """The console's own turns stream live; their history rows must not show twice."""

    def test_console_turn_is_adopted_and_a_second_reply_bubble_too(self, history, pane, feed):
        pane.add_user("dim the kitchen lamp")
        feed.user_submitted("dim the kitchen lamp")
        feed.turn_started()
        first = pane.add_assistant("Dimming.", final=True)
        second = pane.add_assistant("Done, kitchen table lamp at <b>50</b> percent.", final=True)
        now = time.time()
        land(history, entry("Marcus", "dim the kitchen lamp", now, source="Console"),
             entry("assistant", "dim(Köksbord lampa [Kök], 50)\n\nDimming.", now + 0.001))
        land(history, entry("assistant", "Done, kitchen table lamp at <b>50</b> percent.", now + 5))
        assert pane.entries == [pane.entries[0], first, second]   # nothing added while streaming
        feed.turn_finished()
        assert [e.body.text for e in pane.entries] == ["dim the kitchen lamp", "Dimming.",
                                                       "Done, kitchen table lamp at <b>50</b> percent."]

    def test_rows_that_match_no_bubble_are_shown_once_the_reply_is_over(self, history, pane, feed):
        pane.add_user("hello")
        feed.user_submitted("hello")
        feed.turn_started()
        pane.add_assistant("Hi there.", final=True)
        now = time.time()
        land(history, entry("frances", "lights please", now, source="VoiceToVoice", location="Hall"),
             entry("assistant", "Lights on in the hall.", now + 0.001))
        assert [e.kind for e in pane.entries] == [KIND_USER, KIND_ASSISTANT, KIND_USER]  # voice turn shown at once
        feed.turn_finished()
        assert [e.body.text for e in pane.entries] == ["hello", "Hi there.", "lights please", "Lights on in the hall."]

    def test_a_bubble_carrying_a_fallback_notice_adopts_its_row(self, history, pane, feed):
        """The fallback notice is appended to the live reply but kept out of the history,
        so the bubble is longer than its row; it must still count as the same reply."""
        pane.add_user("what other integrations are available")
        feed.user_submitted("what other integrations are available")
        feed.turn_started()
        bubble = pane.add_assistant("Here's what's available.\n\n"
                                    "(Fable refused this request, had to use Opus fallback)", final=True)
        now = time.time()
        land(history, entry("Marcus", "what other integrations are available", now, source="Console"),
             entry("assistant", "Here's what's available.", now + 0.001))
        feed.turn_finished()
        assert pane.entries == [pane.entries[0], bubble]

    def test_a_row_with_inline_code_adopts_its_bubble(self, history, pane, feed):
        """The stream drops the backticks the model wrote; the stored row keeps them."""
        pane.add_user("is it there?")
        feed.user_submitted("is it there?")
        feed.turn_started()
        bubble = pane.add_assistant("Yep, the placeholder  649436  is listed.\n\n"
                                    "(Fable refused this request, had to use Opus fallback)", final=True)
        now = time.time()
        land(history, entry("Marcus", "is it there?", now, source="Console"),
             entry("assistant", "Yep, the placeholder `649436` is listed.", now + 0.001))
        feed.turn_finished()
        assert pane.entries == [pane.entries[0], bubble]

    def test_a_fallback_round_mixing_a_command_with_its_prose_adopts_its_bubble(self, history, pane, feed):
        """The row keeps the command line the stream stripped, the bubble the notice the
        history dropped: neither text contains the other, both must be removed first."""
        pane.add_user("so do the cost tracking analysis please")
        feed.user_submitted("so do the cost tracking analysis please")
        feed.turn_started()
        bubble = pane.add_assistant("Running. Claude Code will report back when it's done.\n\n"
                                    "(Fable refused this request, had to use Opus fallback)", final=True)
        now = time.time()
        land(history, entry("Marcus", "so do the cost tracking analysis please", now, source="Console"),
             entry("assistant", "`self_analysis(\"Cost tracking: where are usage stats stored?\")`\n\n"
                                "Running. Claude Code will report back when it's done.", now + 0.001))
        feed.turn_finished()
        assert pane.entries == [pane.entries[0], bubble]

    def test_a_backticked_command_row_is_hidden(self, history, pane, feed):
        land(history, entry("assistant", "`list_integrations()`", time.time()))
        assert pane.entries == []

    def test_command_only_rows_are_hidden(self, history, pane, feed):
        now = time.time()
        land(history, entry("assistant", 'search_x("M5 Ultra Mac Studio")\ngoogle("Mac Studio M5 news")', now),
             entry("assistant", "<code>open_url(https://example.com/a)</code>", now + 1),
             entry("assistant", "dim(Köksbord lampa [Kök], 50)\n\nDimming.", now + 2))
        assert [e.body.text for e in pane.entries] == ["dim(Köksbord lampa [Kök], 50)\n\nDimming."]

    def test_notifier_broadcast_is_adopted_by_its_history_row(self, history, pane, feed):
        pane.add_notice("Rain in <b>10 minutes</b> — closing the parasol.")
        land(history, entry("assistant", "Rain in 10 minutes — closing the parasol.", time.time()))
        assert [e.kind for e in pane.entries] == [KIND_NOTICE]

    def test_a_bubble_adopts_only_one_row(self, history, pane, feed):
        pane.add_assistant("Done.", final=True)
        now = time.time()
        land(history, entry("assistant", "Done.", now))
        land(history, entry("assistant", "Done.", now + 1))
        assert [e.kind for e in pane.entries] == [KIND_ASSISTANT, KIND_ASSISTANT]

    def test_old_bubbles_are_no_longer_candidates(self, history, pane, feed):
        old = pane.add_assistant("Yesterday's answer.", final=True)
        old.timestamp = time.time() - HistoryFeed.MATCH_WINDOW - 1
        land(history, entry("assistant", "Yesterday's answer.", time.time()))
        assert len(pane.entries) == 2

    def test_console_user_row_without_a_pending_turn_is_shown(self, history, pane, feed):
        land(history, entry("Marcus", "typed in another console", time.time(), source="Console"))
        assert [e.kind for e in pane.entries] == [KIND_USER]

    def test_two_overlapping_turns_settle_together(self, history, pane, feed):
        feed.turn_started()
        feed.turn_started()
        land(history, entry("assistant", "reply", time.time()))
        feed.turn_finished()
        assert pane.entries == []
        feed.turn_finished()
        assert [e.body.text for e in pane.entries] == ["reply"]

    def test_a_turn_the_server_opened_hides_its_brief(self, history, pane, feed):
        # The first-run brief: no user line was typed, the reply streams live, and the
        # history row of the brief itself (role "system", source Console) is not shown.
        feed.turn_started()
        bubble = pane.add_assistant("Hi, I'm Floxy. Let's start with iMessage.", final=True)
        now = time.time()
        land(history, entry("system", "The server has just started…", now, source="Console"),
             entry("assistant", "Hi, I'm Floxy. Let's start with iMessage.", now + 0.001))
        feed.turn_finished()
        assert pane.entries == [bubble]   # the reply row adopted the live bubble, the brief stays hidden

    def test_the_brief_is_shown_when_the_flag_is_on(self, history, pane, feed):
        with patch('interfaces.console.HistoryFeed.SHOW_OPENING_BRIEFS', True):
            land(history, entry("system", "The server has just started…", time.time(), source="Console"))
        assert shown(pane) == [(KIND_SYSTEM, "System (Console)", "The server has just started…")]

class TestIncomingRequests:
    """A voice request is drawn as a system row as it reaches the assistant; the chat
    channels' turns stream live already, so theirs are not."""

    @staticmethod
    def request(source, text, user=None, location=None):
        from assistant.AssistantRequest import RequestContext
        return RequestContext(source=source, user=user, user_input=text, location=location)

    def test_voice_request_lands_at_once_under_a_system_header(self, pane, feed):
        feed.on_request_received(self.request("VoiceToVoice", "turn on the kitchen", user="Marcus",
                                              location="Kitchen"))
        assert shown(pane) == [(KIND_SYSTEM, "System (VoiceToVoice in Kitchen)",
                                "Incoming voice request from Marcus: \u201cturn on the kitchen\u201d")]
        assert pane.entries[0].detail is None

    def test_mic_engine_and_confidence_name_the_origin(self, pane, feed):
        from dataclasses import replace
        ctx = replace(self.request("VoiceToVoice", "turn on the kitchen"), engine="whisper")
        feed.on_request_received(ctx, mic="kitchen", confidence=0.9)
        assert pane.entries[0].body.text == \
            "Incoming voice request (kitchen mic, whisper, c0.90): \u201cturn on the kitchen\u201d"

    def test_a_rejected_transcription_is_shown_with_the_reason_as_detail(self, pane, feed):
        feed.on_request_received(self.request("VoiceToVoice", "turn on the kitchen"), mic="hall", confidence=0.7,
                                 rejected="empty/duplicate/non-best transcription")
        assert pane.entries[0].body.text == "Incoming voice request (hall mic, c0.70): \u201cturn on the kitchen\u201d"
        assert pane.entries[0].detail == "rejected: empty/duplicate/non-best transcription"

    def test_unidentified_speaker_is_left_out(self, pane, feed):
        feed.on_request_received(self.request("VoiceToVoice", "what time is it"))
        assert shown(pane) == [(KIND_SYSTEM, "System (VoiceToVoice)",
                                "Incoming voice request: \u201cwhat time is it\u201d")]

    def test_markup_in_the_query_is_escaped(self, pane, feed):
        feed.on_request_received(self.request("VoiceToVoice", "is 3 < 4 & 5 > 2"))
        assert pane.entries[0].body.text == "Incoming voice request: \u201cis 3 &lt; 4 &amp; 5 &gt; 2\u201d"

    def test_chat_channel_requests_are_not_shown(self, pane, feed):
        feed.on_request_received(self.request("Console", "hello"))
        feed.on_request_received(self.request("Telegram", "hello"))
        assert shown(pane) == []

    def test_the_user_row_still_lands_after_the_exchange(self, history, pane, feed):
        feed.on_request_received(self.request("VoiceToVoice", "turn on the kitchen", user="Marcus"))
        land(history, entry("Marcus", "turn on the kitchen", time.time() + 1, source="VoiceToVoice"))
        assert [k for k, _, _ in shown(pane)] == [KIND_SYSTEM, KIND_USER]


def test_normalize_strips_tags_and_collapses_whitespace():
    assert normalize("  Done,\n<b>kitchen</b>   lamp ") == "Done, kitchen lamp"


class TestWipe:
    def wipe(self, history, remaining):
        """What MessageHistory.wipe_today leaves behind."""
        history.history = list(remaining)
        history.transient_entries = []
        history.wipes += 1
        history._notify_changed()

    def test_the_pane_starts_over_from_what_remains(self, history, pane, feed):
        yesterday = entry("user", "old", T0 - 86400)
        land(history, yesterday, entry("user", "today", T0), entry("assistant", "reply", T0 + 1))
        assert len(pane.entries) == 3
        self.wipe(history, [yesterday])
        assert [e.body.text for e in pane.entries] == ["old"]

    def test_rows_landing_after_a_wipe_are_shown_again(self, history, pane, feed):
        land(history, entry("user", "hi", T0))
        self.wipe(history, [])
        assert pane.entries == []
        land(history, entry("user", "hi", T0))   # same key as the wiped row
        assert [e.body.text for e in pane.entries] == ["hi"]

    def test_a_feed_started_after_a_wipe_does_not_rebuild(self, history, pane):
        history.wipes = 2
        feed = HistoryFeed(history, pane)
        feed.start()
        land(history, entry("user", "hi", T0))
        assert [e.body.text for e in pane.entries] == ["hi"]


class TestCommandResultRows:
    def test_hidden_by_default(self, history, pane, feed):
        land(history, entry("command", "result: \niMessage — disabled", T0 + 30))
        assert shown(pane) == []

    def test_shown_when_enabled(self, history, pane, feed):
        with patch('interfaces.console.HistoryFeed.SHOW_COMMAND_RESULTS', True):
            land(history, entry("command", "result: \niMessage — disabled", T0 + 30))
        assert shown(pane) == [(KIND_SYSTEM, "Command result", "result: \niMessage — disabled")]


class TestClockNoteRows:
    def test_a_byte_prefix_requests_clock_note_is_hidden_but_its_exchange_shows(self, history, pane, feed):
        land(history,
             entry("note", "(Current time: 14:40, Saturday, 2025-03-15)", T0 + 29.999),
             entry("marcus", "what's on today?", T0 + 30, source="Console"),
             entry("assistant", "Nothing booked.", T0 + 30.001))
        assert [(kind, text) for kind, _, text in shown(pane)] == [(KIND_USER, "what's on today?"),
                                                                    (KIND_ASSISTANT, "Nothing booked.")]
