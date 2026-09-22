import json
import sqlite3
import threading
import time

import pytest
from unittest.mock import patch, Mock, MagicMock


def _make_history_section():
    """Create a MessageHistory without the DB load its constructor does."""
    from assistant.prompt.MessageHistory import MessageHistory
    with patch.object(MessageHistory, '_load_history'):
        return MessageHistory()


# ===========================================================================
# TestFormatTimeAgo
# ===========================================================================


# ===========================================================================
# TestAddExchange
# ===========================================================================

class TestAddExchange:
    def test_adds_user_and_assistant(self):
        section = _make_history_section()
        with patch.object(section, '_save_history_entry'), \
             patch.object(section, '_load_history') as mock_load, \
             patch.object(section, '_notify_changed'):
            # Simulate _load_history populating self.history from DB
            def load_side_effect():
                section.history = [
                    {"role": "user", "content": "Hello", "timestamp": 1000.0},
                    {"role": "assistant", "content": "Hi there", "timestamp": 1000.0},
                ]
            mock_load.side_effect = load_side_effect
            section.add_exchange("user", "Hello", "Hi there")
        assert len(section.history) == 2
        assert section.history[0]["role"] == "user"
        assert section.history[0]["content"] == "Hello"
        assert section.history[1]["role"] == "assistant"
        assert section.history[1]["content"] == "Hi there"

    def test_adds_user_only(self):
        section = _make_history_section()
        with patch.object(section, '_save_history_entry'), \
             patch.object(section, '_load_history') as mock_load, \
             patch.object(section, '_notify_changed'):
            def load_side_effect():
                section.history = [{"role": "user", "content": "Hello", "timestamp": 1000.0}]
            mock_load.side_effect = load_side_effect
            section.add_exchange("user", "Hello", None)
        assert len(section.history) == 1
        assert section.history[0]["role"] == "user"

    def test_adds_assistant_only(self):
        section = _make_history_section()
        with patch.object(section, '_save_history_entry'), \
             patch.object(section, '_load_history') as mock_load, \
             patch.object(section, '_notify_changed'):
            def load_side_effect():
                section.history = [{"role": "assistant", "content": "Response", "timestamp": 1000.0}]
            mock_load.side_effect = load_side_effect
            section.add_exchange("user", None, "Response")
        assert len(section.history) == 1
        assert section.history[0]["role"] == "assistant"

    def test_calls_save_entry_load_and_update(self):
        section = _make_history_section()
        with patch.object(section, '_save_history_entry') as mock_save, \
             patch.object(section, '_load_history') as mock_load, \
             patch.object(section, '_notify_changed') as mock_update:
            section.add_exchange("user", "Hello", "Hi")
        assert mock_save.call_count == 2  # once for user, once for assistant
        mock_load.assert_called_once()
        mock_update.assert_called_once()

    def test_save_entry_passes_source_and_location(self):
        section = _make_history_section()
        with patch.object(section, '_save_history_entry') as mock_save, \
             patch.object(section, '_load_history'), \
             patch.object(section, '_notify_changed'):
            section.add_exchange("user", "Hello", "Hi", source="VoiceToVoice", location="Kitchen")
        # First call: user entry with source and location
        user_call = mock_save.call_args_list[0]
        assert user_call[0][0] == "user"
        assert user_call[0][1] == "Hello"
        assert user_call[0][3] == "VoiceToVoice"
        assert user_call[0][4] == "Kitchen"
        # Second call: assistant entry without source/location
        assistant_call = mock_save.call_args_list[1]
        assert assistant_call[0][0] == "assistant"
        assert assistant_call[0][1] == "Hi"


# ===========================================================================
# TestGetMergedEntries
# ===========================================================================

class TestGetMergedEntries:
    def test_empty_returns_empty(self):
        section = _make_history_section()
        assert section.get_merged_entries() == []

    def test_returns_history_sorted_by_timestamp(self):
        section = _make_history_section()
        section.history = [
            {"role": "system", "content": "Light on", "timestamp": 100.0},
            {"role": "user", "content": "Hello", "timestamp": 200.0},
        ]
        entries = section.get_merged_entries()
        assert len(entries) == 2
        assert entries[0]["timestamp"] == 100.0
        assert entries[1]["timestamp"] == 200.0

    def test_respects_max_entries(self):
        section = _make_history_section()
        section.history = [
            {"role": "user", "content": f"msg{i}", "timestamp": float(i)}
            for i in range(700)
        ]
        entries = section.get_merged_entries()
        assert len(entries) == 500

    def test_custom_max_entries(self):
        section = _make_history_section()
        section.history = [
            {"role": "user", "content": f"msg{i}", "timestamp": float(i)}
            for i in range(20)
        ]
        entries = section.get_merged_entries(max_entries=5)
        assert len(entries) == 5


# ===========================================================================
# TestSubscribe
# ===========================================================================

class TestSubscribe:
    def test_subscribers_are_called_after_every_change(self):
        section = _make_history_section()
        calls = []
        section.subscribe(lambda: calls.append(len(section.get_merged_entries())))
        section.add_transient_entry("system", "Prompt section `weather` updated", 1000.0, section="weather")
        assert calls == [1]
        section.clear_transient_entries()
        assert calls == [1, 0]

    def test_a_failing_subscriber_does_not_break_the_others(self):
        section = _make_history_section()
        calls = []

        def boom():
            raise RuntimeError("bad subscriber")

        section.subscribe(boom)
        section.subscribe(lambda: calls.append("ok"))
        section.add_transient_entry("system", "note", 1000.0)
        assert calls == ["ok"]

    def test_unsubscribe(self):
        section = _make_history_section()
        calls = []
        callback = lambda: calls.append(1)
        section.subscribe(callback)
        section.unsubscribe(callback)
        section.add_transient_entry("system", "note", 1000.0)
        assert calls == []


# ===========================================================================
# TestGetHistoryMessages
# ===========================================================================

class TestGetHistoryMessages:
    """History rendered as standard LLM message items (role/content dicts)."""

    def test_empty_returns_empty_list(self):
        section = _make_history_section()
        assert section.get_history_messages() == []

    def test_role_mapping(self):
        from assistant.tools.MessageFormatter import MessageFormatter
        ts = 1000.0
        section = _make_history_section()
        section.history = [
            {"role": "Marcus", "content": "turn off lights", "timestamp": ts,
             "source": "VoiceToVoice", "location": "Kitchen"},
            {"role": "assistant", "content": "Done", "timestamp": ts + 1},
        ]
        msgs = section.get_history_messages()
        assert msgs[0]["role"] == "user"
        # Named-user turn must be byte-identical to the live-query formatting.
        assert msgs[0]["content"] == MessageFormatter.format_message(
            "Marcus", ts, "VoiceToVoice", "Kitchen", "turn off lights")
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Done"

    def test_system_event_rides_on_user_side(self):
        section = _make_history_section()
        section.history = [
            {"role": "system", "content": "Kitchen light on", "timestamp": 1000.0},
            {"role": "Marcus", "content": "hi", "timestamp": 1001.0},
        ]
        msgs = section.get_history_messages()
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"].startswith("System (")
        assert "Kitchen light on" in msgs[0]["content"]

    def test_consecutive_same_role_coalesced(self):
        section = _make_history_section()
        section.history = [
            {"role": "system", "content": "event A", "timestamp": 100.0},
            {"role": "Marcus", "content": "hello", "timestamp": 200.0, "source": None, "location": None},
            {"role": "assistant", "content": "hi", "timestamp": 300.0},
        ]
        msgs = section.get_history_messages()
        # system + named-user both map to "user" and are adjacent → one message.
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert "event A" in msgs[0]["content"]
        assert "hello" in msgs[0]["content"]
        assert msgs[1]["role"] == "assistant"

    def test_leading_non_user_dropped(self):
        section = _make_history_section()
        section.history = [
            {"role": "assistant", "content": "autonomous note", "timestamp": 100.0},
            {"role": "Marcus", "content": "hello", "timestamp": 200.0, "source": None, "location": None},
        ]
        msgs = section.get_history_messages()
        assert msgs[0]["role"] == "user"
        assert all("autonomous note" not in m["content"] for m in msgs)

    def test_byte_budget_drops_oldest(self):
        section = _make_history_section()
        section.max_render_chars = 10_000
        section.history = [
            {"role": "Marcus", "content": "A" * 150, "timestamp": float(i),
             "source": None, "location": None}
            for i in range(10)
        ]
        msgs = section.get_history_messages(max_bytes=400)
        total = sum(len(m["content"].encode("utf-8")) for m in msgs)
        assert total <= 400
        # Newest entry preserved.
        assert any("A" * 150 in m["content"] for m in msgs)

    def test_per_entry_truncation(self):
        section = _make_history_section()
        section.max_render_chars = 100
        section.history = [
            {"role": "Marcus", "content": "x" * 250, "timestamp": 1000.0,
             "source": None, "location": None},
        ]
        msgs = section.get_history_messages()
        assert "x" * 100 + "..." in msgs[0]["content"]
        assert "x" * 101 not in msgs[0]["content"]

    def test_assistant_truncation_keeps_tail(self):
        """Long assistant turns are middle-cut so the user-visible reply (after the
        thinking block) survives — a head-only cut left a trailing "..." the model
        misread as its own unfinished message, causing duplicate notifications."""
        section = _make_history_section()
        section.max_render_chars = 100
        thinking = "<thinking>" + "t" * 200 + "</thinking>"
        reply = "The garage door is still open, close it before bed."
        section.history = [
            {"role": "Marcus", "content": "status?", "timestamp": 1000.0,
             "source": None, "location": None},
            {"role": "assistant", "content": thinking + "\n" + reply, "timestamp": 1001.0},
        ]
        msgs = section.get_history_messages()
        assistant = next(m for m in msgs if m["role"] == "assistant")
        assert assistant["content"].endswith(reply[-25:])
        assert " […history truncated…] " in assistant["content"]
        assert not assistant["content"].endswith("...")

    def test_assistant_truncation_respects_cap(self):
        section = _make_history_section()
        section.max_render_chars = 100
        section.history = [
            {"role": "Marcus", "content": "hi", "timestamp": 1000.0,
             "source": None, "location": None},
            {"role": "assistant", "content": "a" * 75 + "b" * 25 + "c" * 400, "timestamp": 1001.0},
        ]
        msgs = section.get_history_messages()
        assistant = next(m for m in msgs if m["role"] == "assistant")
        # head = 75 chars, tail = 25 chars (cap 100), plus the marker between them.
        assert assistant["content"] == "a" * 75 + " […history truncated…] " + "c" * 25

    def test_assistant_at_cap_not_truncated(self):
        section = _make_history_section()
        section.max_render_chars = 100
        exact = "z" * 100
        section.history = [
            {"role": "Marcus", "content": "hi", "timestamp": 1000.0,
             "source": None, "location": None},
            {"role": "assistant", "content": exact, "timestamp": 1001.0},
        ]
        msgs = section.get_history_messages()
        assistant = next(m for m in msgs if m["role"] == "assistant")
        assert assistant["content"] == exact


# ===========================================================================
# TestHysteresisTrim
# ===========================================================================

class TestHysteresisTrim:
    """get_history_messages trims with hysteresis: the window grows to
    trim_high_bytes, is cut back to trim_low_bytes, and the head stays frozen
    (sticky _trim_cutoff) until the high watermark is hit again — so the
    rendered prefix, and with it the message-level KV cache, survives turns."""

    @staticmethod
    def _entries(n, size=100, start_ts=1000.0):
        return [
            {"role": "Marcus", "content": f"m{i:03d}" + "x" * size, "timestamp": start_ts + i,
             "source": None, "location": None}
            for i in range(n)
        ]

    def test_under_high_watermark_keeps_everything(self):
        section = _make_history_section()
        section.trim_high_bytes = 10_000
        section.trim_low_bytes = 5_000
        section.history = self._entries(10)
        joined = "\n".join(m["content"] for m in section.get_history_messages())
        assert "m000" in joined and "m009" in joined
        assert section._trim_cutoff == 0.0

    def test_over_high_trims_to_low_watermark(self):
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = self._entries(30)
        msgs = section.get_history_messages()
        joined = "\n".join(m["content"] for m in msgs)
        assert "m029" in joined, "newest entry must survive the trim"
        assert "m000" not in joined, "oldest entries must be trimmed"
        # Window content (minus coalescing separators) must respect the low watermark.
        assert len(joined.encode("utf-8")) <= 1_000 + len(msgs) * 2
        assert section._trim_cutoff > 1000.0, "cutoff must advance on trim"

    def test_head_stays_frozen_between_watermarks(self):
        """The whole point: new turns arriving while the window is between LOW and
        HIGH must not move the head — the rendered prefix stays byte-identical."""
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = self._entries(30)
        first = section.get_history_messages()
        cutoff_after_trim = section._trim_cutoff

        section.history.append(
            {"role": "Marcus", "content": "a new question", "timestamp": 2000.0,
             "source": None, "location": None})
        second = section.get_history_messages()

        assert section._trim_cutoff == cutoff_after_trim, "cutoff must not move under HIGH"
        prefix = first[0]["content"][:80]
        assert second[0]["content"].startswith(prefix), "window head must stay byte-stable"


    def test_trim_window_moves_the_cutoff_where_the_next_window_would_put_it(self):
        """ToolkitPreloader asks trim_window() before deciding what the window holds:
        the cutoff must land where get_history_messages() would put it, and stay there —
        otherwise a toolkit it saw in the window is trimmed away by the messages that follow."""
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = self._entries(5)
        section.trim_window()
        assert section._trim_cutoff == 0.0, "under the high watermark there is nothing to trim"

        section.history = self._entries(30)
        section.trim_window()
        cutoff = section._trim_cutoff
        assert cutoff > 1000.0, "over the high watermark the cutoff advances before any messages are built"
        joined = "\n".join(m["content"] for m in section.get_history_messages())
        assert section._trim_cutoff == cutoff, "the messages that follow use the same window"
        assert "m000" not in joined and "m029" in joined

    def test_window_starts_at_user_turn_after_trim(self):
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        entries = []
        for i in range(20):
            entries.append({"role": "Marcus", "content": f"q{i:03d}" + "x" * 80,
                            "timestamp": 1000.0 + i * 2, "source": None, "location": None})
            entries.append({"role": "assistant", "content": f"a{i:03d}" + "y" * 80,
                            "timestamp": 1001.0 + i * 2})
        section.history = entries
        msgs = section.get_history_messages()
        assert msgs[0]["role"] == "user"
        # The first assistant reply in the window must have its user query present:
        # the window head snaps forward to a user-side turn, never an orphaned reply.
        first_reply = msgs[1]["content"][:4]
        assert first_reply.replace("a", "q") in msgs[0]["content"]

    @staticmethod
    def _tied_batch(n, ts=1500.0, size=100):
        """A multi-photo iMessage batch: one history row per attachment, all with
        the same timestamp (chat.db stores one date for the whole send)."""
        return [
            {"role": "system", "content": f"photo{i:02d}" + "p" * size, "timestamp": ts,
             "source": "iMessage", "location": None}
            for i in range(n)
        ]

    @staticmethod
    def _trim_logs(section):
        with patch("assistant.prompt.MessageHistory.info") as mock_info:
            msgs = section.get_history_messages()
        logs = [str(c) for c in mock_info.call_args_list if "window hit" in str(c)]
        return msgs, logs

    def test_trim_never_cuts_inside_a_timestamp_tie(self):
        """The cutoff is a bare timestamp, so a cut inside a tied batch would be
        undone by the `>=` filter on the next call — the trim would fire (and log)
        on every single call forever. The head must snap back to the tie start."""
        section = _make_history_section()
        section.trim_high_bytes = 3_000
        section.trim_low_bytes = 2_000
        # Low-watermark walk from the newest entry lands inside the photo batch.
        section.history = self._entries(10) + self._tied_batch(12) + self._entries(5, start_ts=1600.0)

        first, logs = self._trim_logs(section)
        assert len(logs) == 1
        assert section._trim_cutoff == 1500.0, "the head must land on the tied batch's timestamp"
        joined = "\n".join(m["content"] for m in first)
        assert all(f"photo{i:02d}" in joined for i in range(12)), "tied batch kept whole"
        assert "m009" not in joined

        second, logs = self._trim_logs(section)
        assert second == first, "a second call must be a no-op with an identical window"
        assert not logs, "no re-trim (and no log line) once the window has converged"

    def test_tie_too_big_to_keep_is_cut_past_in_one_step(self):
        """When keeping the tied batch whole would leave the window over HIGH, the
        trim must move past the batch right away — not park at an over-HIGH window
        and re-trim on the next call."""
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = self._entries(10) + self._tied_batch(12) + self._entries(5, start_ts=1600.0)

        first, logs = self._trim_logs(section)
        assert len(logs) == 1 and section._trim_cutoff == 1600.0
        joined = "\n".join(m["content"] for m in first)
        assert "photo00" not in joined and "m000" in joined and "m004" in joined
        second, logs = self._trim_logs(section)
        assert second == first and not logs

    def test_tie_starting_with_an_assistant_reply_still_converges(self):
        """An assistant reply sharing its timestamp with the next user turn: pulling the
        tie back puts an orphaned reply at the window head. _finalize_messages pops it,
        and the cutoff must still hold on the next call instead of re-trimming."""
        section = _make_history_section()
        section.trim_high_bytes = 3_000
        section.trim_low_bytes = 2_000
        tie = [{"role": "assistant", "content": "a-tied" + "y" * 100, "timestamp": 1500.0}] + [
            {"role": "Marcus", "content": f"q-tied{i:02d}" + "x" * 100, "timestamp": 1500.0,
             "source": None, "location": None} for i in range(11)]
        section.history = self._entries(10) + tie + self._entries(5, start_ts=1600.0)

        first, logs = self._trim_logs(section)
        assert len(logs) == 1 and section._trim_cutoff == 1500.0
        assert first[0]["role"] == "user" and "q-tied00" in first[0]["content"]
        assert "a-tied" not in "\n".join(m["content"] for m in first)
        second, logs = self._trim_logs(section)
        assert second == first and not logs and section._trim_cutoff == 1500.0

    def test_all_entries_tied_is_left_alone_without_logging(self):
        section = _make_history_section()
        section.trim_high_bytes = 500
        section.trim_low_bytes = 200
        section.history = self._tied_batch(12)
        first, logs = self._trim_logs(section)
        joined = "\n".join(m["content"] for m in first)
        assert all(f"photo{i:02d}" in joined for i in range(12))
        assert not logs and section._trim_cutoff == 0.0

    def test_explicit_max_bytes_is_one_shot_and_keeps_cutoff(self):
        section = _make_history_section()
        section.max_render_chars = 10_000
        section.history = self._entries(30)
        msgs = section.get_history_messages(max_bytes=400)
        total = sum(len(m["content"].encode("utf-8")) for m in msgs)
        assert total <= 400
        assert section._trim_cutoff == 0.0, "explicit budget must not touch the sticky window"

    def test_entries_older_than_cutoff_stay_hidden(self):
        """Even if the window has free space again, entries behind the cutoff must
        not reappear — reappearing entries would shift the head backwards."""
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = self._entries(30)
        section.get_history_messages()  # forces a trim, advances cutoff

        # Raise the watermark so everything would fit now.
        section.trim_high_bytes = 100_000
        joined = "\n".join(m["content"] for m in section.get_history_messages())
        assert "m000" not in joined


# ===========================================================================
# TestLoadSaveHistory
# ===========================================================================

class TestLoadSaveHistory:
    def test_load_reads_from_database(self):
        section = _make_history_section()
        db_rows = [
            {"role": "user", "content": "Hello", "timestamp": 100.0, "source": None, "location": None}
        ]
        mock_db = MagicMock()
        mock_db.get_recent_history.return_value = db_rows
        with patch('assistant.prompt.MessageHistory.FloxyDatabase') as MockDB:
            MockDB.get_instance.return_value = mock_db
            section._load_history()
        assert len(section.history) == 1
        assert section.history[0]["content"] == "Hello"
        mock_db.get_recent_history.assert_called_once_with(500)

    def test_load_db_error_returns_empty(self):
        section = _make_history_section()
        with patch('assistant.prompt.MessageHistory.FloxyDatabase') as MockDB:
            MockDB.get_instance.side_effect = Exception("DB error")
            section._load_history()
        assert section.history == []

    def test_save_entry_writes_to_database(self):
        section = _make_history_section()
        mock_db = MagicMock()
        with patch('assistant.prompt.MessageHistory.FloxyDatabase') as MockDB:
            MockDB.get_instance.return_value = mock_db
            section._save_history_entry("user", "Hello", 100.0, "VoiceToVoice", "Kitchen")
        mock_db.add_history_entry.assert_called_once_with("user", "Hello", 100.0, "VoiceToVoice", "Kitchen")


# ===========================================================================
# TestRenderForDisplay
# ===========================================================================

class TestRenderForDisplay:
    """render_for_display() powers the console PromptTab viewer."""

    def test_empty_history_returns_placeholder(self):
        section = _make_history_section()
        section.history = []
        assert section.render_for_display() == "_No conversation history yet._"

    def test_renders_all_entries_as_markdown_headers(self):
        from datetime import datetime
        ts = datetime(2025, 3, 15, 14, 40, 0).timestamp()
        section = _make_history_section()
        section.history = [
            {"role": "Marcus", "content": "turn off the lights", "timestamp": ts,
             "source": "VoiceToVoice", "location": "Kitchen"},
            {"role": "assistant", "content": "Done.", "timestamp": ts},
            {"role": "system", "content": "iMessage received: hi", "timestamp": ts},
        ]
        out = section.render_for_display()
        assert "### Marcus (14:40, VoiceToVoice in Kitchen)" in out
        assert "turn off the lights" in out
        assert "### Assistant (14:40)" in out
        assert "### System (14:40)" in out
        assert "iMessage received: hi" in out

    def test_shows_full_content_without_truncation(self):
        section = _make_history_section()
        section.max_render_chars = 50  # the LLM rendering would truncate here; this must not.
        long_text = "z" * 500
        section.history = [{"role": "user", "content": long_text, "timestamp": 1000.0}]
        assert long_text in section.render_for_display()

    def test_hides_entries_before_trim_cutoff_with_note(self):
        section = _make_history_section()
        section.history = [
            {"role": "user", "content": "old question", "timestamp": 1000.0},
            {"role": "assistant", "content": "old answer", "timestamp": 1001.0},
            {"role": "user", "content": "new question", "timestamp": 2000.0},
        ]
        section._trim_cutoff = 2000.0
        out = section.render_for_display()
        assert "old question" not in out
        assert "old answer" not in out
        assert "new question" in out
        assert out.startswith("_2 older entries (before ")
        assert "trimmed from the window" in out


# ===========================================================================
# TestLoadFixtureMessages
# ===========================================================================

class TestLoadFixtureMessages:
    """load_fixture_messages parses the prompt test suite's JSON history fixture into
    LLM message turns, converting the 'YYYY-MM-DD HH:MM:SS' timestamp to Unix time and
    rendering byte-identically to live history so the suite sends the exact same turns."""

    def test_parses_fixture_into_message_turns(self, tmp_path):
        from datetime import datetime
        from assistant.prompt.MessageHistory import MessageHistory
        from assistant.tools.MessageFormatter import MessageFormatter

        fixture = [
            {"role": "Marcus", "content": "turn off lights",
             "timestamp": "2026-05-20 14:40:00", "source": "VoiceToVoice", "location": "Kitchen"},
            {"role": "assistant", "content": "Done.", "timestamp": "2026-05-20 14:40:05"},
        ]
        path = tmp_path / "history.json"
        path.write_text(json.dumps(fixture))

        msgs = MessageHistory.load_fixture_messages(str(path))

        ts = datetime(2026, 5, 20, 14, 40, 0).timestamp()
        assert msgs[0]["role"] == "user"
        # Timestamp string must be parsed to the right Unix time and rendered byte-identically
        # to a live named-user turn (so the prompt cache stays warm across runs).
        assert msgs[0]["content"] == MessageFormatter.format_message(
            "Marcus", ts, "VoiceToVoice", "Kitchen", "turn off lights")
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Done."

    def test_empty_fixture_returns_empty_list(self, tmp_path):
        from assistant.prompt.MessageHistory import MessageHistory
        path = tmp_path / "empty.json"
        path.write_text("[]")
        assert MessageHistory.load_fixture_messages(str(path)) == []

    def test_blank_turns_are_dropped(self):
        """An assistant reply interrupted by a restart can be stored empty; it must not
        re-enter the conversation (and would crash Claude's empty-block cache breakpoint)."""
        from datetime import datetime
        from assistant.prompt.MessageHistory import MessageHistory

        entries = [
            {"role": "Marcus", "content": "what's the weather",
             "timestamp": datetime(2026, 5, 20, 9, 15, 0).timestamp(), "source": None, "location": None},
            {"role": "assistant", "content": "   ",
             "timestamp": datetime(2026, 5, 20, 9, 15, 3).timestamp()},
            {"role": "Marcus", "content": "hej",
             "timestamp": datetime(2026, 5, 20, 9, 16, 0).timestamp(), "source": None, "location": None},
        ]
        msgs = MessageHistory.render_messages(
            entries, 20 * 1024, MessageHistory.MAX_RENDER_CHARS)

        assert all(m["content"].strip() for m in msgs)
        assert all(m["role"] == "user" for m in msgs)

    def test_renders_identically_to_live_history(self, tmp_path):
        """The fixture path and the live-history path must produce byte-identical turns —
        both go through render_messages, so equivalent entries must come out the same."""
        from datetime import datetime
        from assistant.prompt.MessageHistory import MessageHistory

        fixture = [
            {"role": "Marcus", "content": "what's the weather",
             "timestamp": "2026-05-20 09:15:00", "source": None, "location": None},
            {"role": "assistant", "content": "Sunny.", "timestamp": "2026-05-20 09:15:03"},
        ]
        path = tmp_path / "history.json"
        path.write_text(json.dumps(fixture))

        from_fixture = MessageHistory.load_fixture_messages(str(path))

        equivalent_entries = [
            {"role": "Marcus", "content": "what's the weather",
             "timestamp": datetime(2026, 5, 20, 9, 15, 0).timestamp(), "source": None, "location": None},
            {"role": "assistant", "content": "Sunny.",
             "timestamp": datetime(2026, 5, 20, 9, 15, 3).timestamp()},
        ]
        from_live = MessageHistory.render_messages(
            equivalent_entries, 20 * 1024, MessageHistory.MAX_RENDER_CHARS)

        assert from_fixture == from_live


# ===========================================================================
# TestTransientEntries
# ===========================================================================

class TestTransientEntries:
    """Transient entries are in-memory-only history rows used by PromptManager to
    inject 'system prompt diff' notes between full prompt rebuilds. They must
    render alongside persisted history but never hit floxy.db."""

    def test_add_transient_entry_does_not_write_to_db(self):
        section = _make_history_section()
        with patch('assistant.prompt.MessageHistory.FloxyDatabase') as MockDB:
            section.add_transient_entry("system", "diff note", 1000.0, source="prompt_delta")
            MockDB.get_instance.assert_not_called()
        assert len(section.transient_entries) == 1
        assert section.transient_entries[0]["content"] == "diff note"
        assert section.transient_entries[0]["source"] == "prompt_delta"

    def test_transient_entry_appears_in_get_history_messages(self):
        section = _make_history_section()
        section.history = [
            {"role": "Marcus", "content": "hi", "timestamp": 1000.0,
             "source": None, "location": None},
            {"role": "assistant", "content": "hello", "timestamp": 1000.5},
        ]
        section.add_transient_entry("system", "Light turned on", 1001.0, source="prompt_delta")
        msgs = section.get_history_messages()
        # transient gets rendered as user-side "System (HH:MM): ..." turn
        joined = "\n".join(m["content"] for m in msgs)
        assert "Light turned on" in joined

    def test_clear_transient_entries_drops_them_from_rendering(self):
        section = _make_history_section()
        section.add_transient_entry("system", "delta one", 1000.0)
        section.add_transient_entry("system", "delta two", 1001.0)
        assert len(section.transient_entries) == 2

        section.clear_transient_entries()
        assert section.transient_entries == []

        msgs = section.get_history_messages()
        assert msgs == []

    def test_transient_and_persisted_merge_sorted_by_timestamp(self):
        section = _make_history_section()
        section.history = [
            {"role": "Marcus", "content": "first", "timestamp": 1000.0,
             "source": None, "location": None},
            {"role": "assistant", "content": "third", "timestamp": 1002.0},
        ]
        section.add_transient_entry("system", "second", 1001.0)

        merged = section.get_merged_entries()
        contents = [e["content"] for e in merged]
        assert contents == ["first", "second", "third"]

    def test_add_transient_entry_stores_section_tag(self):
        section = _make_history_section()
        section.add_transient_entry("system", "weather diff", 1000.0,
                                    source="prompt_delta", section="weather")
        assert section.transient_entries[0]["section"] == "weather"

    def test_clear_transient_entries_for_section_drops_only_matching(self):
        section = _make_history_section()
        section.add_transient_entry("system", "weather diff", 1000.0, section="weather")
        section.add_transient_entry("system", "calendar diff", 1001.0, section="calendar")
        section.add_transient_entry("system", "weather diff 2", 1002.0, section="weather")

        dropped = section.clear_transient_entries_for_section("weather")

        assert dropped == 2
        remaining = section.transient_entries
        assert len(remaining) == 1
        assert remaining[0]["section"] == "calendar"

    def test_clear_transient_entries_for_section_noop_when_no_match(self):
        section = _make_history_section()
        section.add_transient_entry("system", "calendar diff", 1000.0, section="calendar")

        dropped = section.clear_transient_entries_for_section("weather")

        assert dropped == 0
        assert len(section.transient_entries) == 1

    def test_clear_transient_entries_for_section_ignores_untagged(self):
        """Untagged transients (section=None) must not be swept by a section-scoped
        clear — only the full clear_transient_entries() removes them."""
        section = _make_history_section()
        section.add_transient_entry("system", "untagged note", 1000.0)
        assert section.transient_entries[0]["section"] is None

        dropped = section.clear_transient_entries_for_section("weather")

        assert dropped == 0
        assert len(section.transient_entries) == 1

    def test_clear_transient_entries_keeps_the_carried_sections_notes(self):
        """A full prompt rebuild bakes every section but a carried one (the Indigo data rides
        with the home toolkit), so its notes survive the sweep while the rest go."""
        section = _make_history_section()
        section.add_transient_entry("system", "weather diff", 1000.0, section="weather")
        section.add_transient_entry("system", "indigo diff", 1001.0, section="indigo")
        section.add_transient_entry("system", "untagged note", 1002.0)
        with patch.object(section, '_notify_changed') as notify:
            section.clear_transient_entries(keep_sections=("indigo",))
        assert [e["content"] for e in section.transient_entries] == ["indigo diff"]
        assert notify.call_count == 1

    def test_clear_transient_entries_with_only_kept_notes_changes_nothing(self):
        section = _make_history_section()
        section.add_transient_entry("system", "indigo diff", 1001.0, section="indigo")
        with patch.object(section, '_notify_changed') as notify:
            section.clear_transient_entries(keep_sections=("indigo",))
        assert len(section.transient_entries) == 1
        notify.assert_not_called()


# ===========================================================================
# TestAppendUserTurn
# ===========================================================================

class TestAppendUserTurn:
    """append_user_turn is the shared "tack a user-side note onto the trailing turn"
    helper used by PromptManager.get_messages (clock + thinking instruction) and
    AutonomousActionManager.activity_check (clock + AUTONOMOUS_ACTION_PROMPT).

    Two invariants must hold across every call path or the LLM request breaks:
      1. Consecutive same-role turns must never appear — Gemini and other providers
         reject them outright. So when the trailing turn is already user, the new
         content must merge in rather than appending a second user turn.
      2. The merge guard must only fire on string content. A multimodal user turn
         (vision: content is a list of text/image blocks) must NOT be rewritten as
         a string — that would corrupt the image blocks and the LLM request would
         either reject the payload or silently lose the image.
    """

    def test_appends_when_messages_empty(self):
        from assistant.prompt.MessageHistory import MessageHistory
        messages = []
        MessageHistory.append_user_turn(messages, "hello")
        assert messages == [{"role": "user", "content": "hello"}]

    def test_appends_after_assistant_turn(self):
        from assistant.prompt.MessageHistory import MessageHistory
        messages = [
            {"role": "user", "content": "what time is it?"},
            {"role": "assistant", "content": "3pm"},
        ]
        MessageHistory.append_user_turn(messages, "thanks")
        # New user turn after assistant — no merge possible, must append.
        assert len(messages) == 3
        assert messages[-1] == {"role": "user", "content": "thanks"}

    def test_merges_into_trailing_user_turn_with_double_newline(self):
        """When the trailing turn is already user with string content, the new note
        must merge into it. The "\\n\\n" separator is the LLM-facing format the system
        and the prompt cache have been calibrated against — a silent format drift
        (e.g. single newline, or no separator) would degrade time/instruction
        attention without anything failing loudly."""
        from assistant.prompt.MessageHistory import MessageHistory
        messages = [
            {"role": "user", "content": "Bob (iMessage): you up?"},
        ]
        MessageHistory.append_user_turn(messages, "(Current time: 14:50)")
        assert len(messages) == 1, "must merge, not append a second user turn"
        assert messages[0] == {
            "role": "user",
            "content": "Bob (iMessage): you up?\n\n(Current time: 14:50)",
        }

    def test_does_not_merge_into_multimodal_user_turn(self):
        """Regression guard: vision turns hold list content (text + image blocks).
        Merging by concatenation would either crash (str + list) or, if the guard
        was loosened, silently rewrite the turn to a string and drop the image
        blocks. Either way the LLM request is broken — the new note must land as
        its own trailing turn instead."""
        from assistant.prompt.MessageHistory import MessageHistory
        image_content = [
            {"type": "text", "text": "what's in this photo?"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ]
        messages = [{"role": "user", "content": image_content}]
        MessageHistory.append_user_turn(messages, "(Current time: 14:50)")

        # The multimodal turn must be untouched, the note appended as its own turn.
        assert len(messages) == 2
        assert messages[0]["content"] is image_content, "multimodal content must not be rewritten"
        assert messages[1] == {"role": "user", "content": "(Current time: 14:50)"}

    def test_successive_appends_merge_into_single_user_turn(self):
        """Realistic call pattern: get_messages and activity_check both call
        append_user_turn twice in a row (clock, then instruction). After history
        ends in assistant, the first call appends a new user turn; the second
        must merge into it — not append a third turn."""
        from assistant.prompt.MessageHistory import MessageHistory
        messages = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "ok"},
        ]
        MessageHistory.append_user_turn(messages, "(Current time: 14:50)")
        MessageHistory.append_user_turn(messages, "INSTRUCTION")

        assert len(messages) == 3, "two consecutive appends must collapse into one trailing user turn"
        assert messages[-1] == {
            "role": "user",
            "content": "(Current time: 14:50)\n\nINSTRUCTION",
        }


# ===========================================================================
# TestWipeToday
# ===========================================================================

class TestWipeToday:
    def test_deletes_todays_rows_and_starts_over(self):
        section = _make_history_section()
        section.transient_entries = [{"role": "system", "content": "delta", "timestamp": 5.0, "section": "x"}]
        section._trim_cutoff = 4.0
        notified = []
        section.subscribe(lambda: notified.append(True))
        db = Mock()
        db.delete_todays_history.return_value = 3
        db.get_recent_history.return_value = [{"role": "user", "content": "old", "timestamp": 1.0}]
        with patch('assistant.prompt.MessageHistory.FloxyDatabase.get_instance', return_value=db):
            deleted = section.wipe_today()
        assert deleted == 3
        assert section.history == [{"role": "user", "content": "old", "timestamp": 1.0}]
        assert section.transient_entries == []
        assert section._trim_cutoff == 0.0
        assert section.wipes == 1
        assert notified == [True]
        assert section.last_update_time > 0


class TestCommandResults:
    """A command round's result is persisted as a `command` row so the next turn still
    sees it, and renders as the very `result:` user turn the follow-up round was sent."""

    def _section_with(self, rows):
        section = _make_history_section()
        section.history = list(rows)
        return section

    def test_persisted_after_the_newest_row(self):
        section = self._section_with([
            {"role": "assistant", "content": "list_integrations()", "timestamp": time.time() + 100},
        ])
        with patch.object(section, '_save_history_entry') as save, \
             patch.object(section, '_load_history'), \
             patch.object(section, '_notify_changed') as notify:
            section.add_command_result("result: \niMessage — disabled")
        role, content, timestamp = save.call_args.args[:3]
        assert role == "command"
        assert content == "result: \niMessage — disabled"
        assert timestamp > section.history[0]["timestamp"]
        assert notify.called

    def test_empty_result_not_persisted(self):
        section = self._section_with([])
        with patch.object(section, '_save_history_entry') as save, \
             patch.object(section, '_load_history'), patch.object(section, '_notify_changed'):
            section.add_command_result("   ")
        assert not save.called

    def test_oversized_result_capped(self):
        section = self._section_with([])
        with patch.object(section, '_save_history_entry') as save, \
             patch.object(section, '_load_history'), patch.object(section, '_notify_changed'):
            section.add_command_result("x" * (section.COMMAND_RESULT_CHARS + 500))
        content = save.call_args.args[1]
        assert content.startswith("x" * section.COMMAND_RESULT_CHARS)
        assert content.endswith("[…result truncated in history…]")

    def test_uncapped_result_stored_whole(self):
        section = self._section_with([])
        with patch.object(section, '_save_history_entry') as save, \
             patch.object(section, '_load_history'), patch.object(section, '_notify_changed'):
            section.add_command_result("x" * (section.COMMAND_RESULT_CHARS + 500), cap=False)
        assert save.call_args.args[1] == "x" * (section.COMMAND_RESULT_CHARS + 500)

    def test_loaded_toolkits_are_the_toolkit_rows_inside_the_window(self):
        section = self._section_with([
            {"role": "command", "content": "result: \n# Toolkit: web\ntext", "timestamp": 50.0},
            {"role": "command", "content": "result: \n# Toolkit: email\ntext\n\n# Toolkit: x\ntext",
             "timestamp": 100.0},
            {"role": "command", "content": "result: \nplain result", "timestamp": 101.0},
            {"role": "assistant", "content": "# Toolkit: music", "timestamp": 102.0},
        ])
        section._trim_cutoff = 60.0     # the web row was trimmed out of the window
        assert section.loaded_toolkits() == {"email": 100.0, "x": 100.0}

    def test_loaded_toolkits_empty_without_rows(self):
        assert self._section_with([]).loaded_toolkits() == {}

    def test_renders_verbatim_on_the_user_side(self):
        big = "result: \n" + "line\n" * 1000     # far past MAX_RENDER_CHARS, stays whole
        section = self._section_with([
            {"role": "Marcus", "content": "enable imessage", "timestamp": 100.0, "source": "Console"},
            {"role": "assistant", "content": "list_integrations()", "timestamp": 100.001},
            {"role": "command", "content": big, "timestamp": 100.002},
            {"role": "assistant", "content": "Done, it is on.", "timestamp": 100.003},
        ])
        msgs = section.get_history_messages()
        assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
        assert msgs[2]["content"] == big

    def test_markdown_view_labels_the_row(self):
        section = self._section_with([
            {"role": "Marcus", "content": "hi", "timestamp": 100.0, "source": "Console"},
            {"role": "command", "content": "result: \nok", "timestamp": 100.002},
        ])
        text = section.render_for_display()
        assert "### Command result (" in text
        assert "result: \nok" in text


class TestAddContextMessages:
    def test_a_failed_batch_write_propagates_without_reloading(self):
        from assistant.integration.Feed import FeedEntry
        section = _make_history_section()
        db = Mock()
        db.add_history_entries.side_effect = sqlite3.OperationalError("database is locked")
        with patch('assistant.prompt.MessageHistory.FloxyDatabase.get_instance', return_value=db), \
             patch.object(section, '_load_history') as load, \
             patch.object(section, '_notify_changed') as notify:
            with pytest.raises(sqlite3.OperationalError):
                section.add_context_messages("iMessage", [FeedEntry(1.0, "hi")])
        load.assert_not_called()
        notify.assert_not_called()

    def test_a_stored_batch_reloads_and_notifies_once(self):
        from assistant.integration.Feed import FeedEntry
        section = _make_history_section()
        db = Mock()
        db.add_history_entries.return_value = 2
        with patch('assistant.prompt.MessageHistory.FloxyDatabase.get_instance', return_value=db), \
             patch.object(section, '_load_history') as load, \
             patch.object(section, '_notify_changed') as notify:
            stored = section.add_context_messages("iMessage", [FeedEntry(1.0, "a"), FeedEntry(2.0, "b")])
        assert stored == 2
        assert load.call_count == 1
        assert notify.call_count == 1

    def test_an_all_duplicate_batch_leaves_the_history_untouched(self):
        """The Indigo feed re-submits the whole day every poll and relies on the dedup index;
        a batch that stores nothing must not bump last_update_time or wake the subscribers,
        or the dashboard's History row resets every poll for nothing."""
        from assistant.integration.Feed import FeedEntry
        section = _make_history_section()
        db = Mock()
        db.add_history_entries.return_value = 0
        with patch('assistant.prompt.MessageHistory.FloxyDatabase.get_instance', return_value=db), \
             patch.object(section, '_load_history') as load, \
             patch.object(section, '_notify_changed') as notify:
            stored = section.add_context_messages("Indigo", [FeedEntry(1.0, "a"), FeedEntry(2.0, "b")])
        assert stored == 0
        db.add_history_entries.assert_called_once()
        load.assert_not_called()
        notify.assert_not_called()


# ===========================================================================
# TestToolkitEvictionStubs — a trim that drops a toolkit row leaves a stub in its place
# ===========================================================================

class TestToolkitEvictionStubs:
    STUB = 'load_toolkit("browser")'

    @staticmethod
    def _turns(n, start_ts, size=100):
        return [{"role": "Marcus", "content": f"m{i:03d}" + "x" * size, "timestamp": start_ts + i,
                 "source": None, "location": None} for i in range(n)]

    @staticmethod
    def _toolkit_row(name, timestamp, size=800):
        return {"role": "command", "content": f"result: \n# Toolkit: {name}\n" + "z" * size,
                "timestamp": timestamp, "source": None, "location": None}

    def _section_after_eviction(self):
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = [
            {"role": "Marcus", "content": "open ellos", "timestamp": 1000.0, "source": None, "location": None},
            {"role": "assistant", "content": 'load_toolkit("browser")', "timestamp": 1001.0},
            self._toolkit_row("browser", 1002.0),
        ] + self._turns(20, 1010.0)
        return section, section.get_history_messages()

    @staticmethod
    def _joined(msgs):
        return "\n".join(m["content"] for m in msgs)

    def test_the_stub_stands_where_the_evicted_row_was(self):
        section, msgs = self._section_after_eviction()
        joined = self._joined(msgs)
        assert "# Toolkit: browser" not in joined, "the reference itself is gone"
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"].startswith("System (")
        assert "browser toolkit reference" in msgs[0]["content"] and self.STUB in msgs[0]["content"]
        assert joined.count(self.STUB) == 1
        assert section._trim_cutoff > 1002.0, "the stub is exempt from the cutoff, not the cutoff"

    def test_the_stub_is_not_a_loaded_toolkit(self):
        section, _ = self._section_after_eviction()
        assert section.loaded_toolkits() == {}

    def test_rendering_again_keeps_one_stub(self):
        section, _ = self._section_after_eviction()
        joined = self._joined(section.get_history_messages())
        assert joined.count(self.STUB) == 1
        assert len(section.transient_entries) == 1

    def test_the_stub_retires_when_the_toolkit_is_loaded_again(self):
        section, _ = self._section_after_eviction()
        section.history.append(self._toolkit_row("browser", 2000.0, size=10))
        joined = self._joined(section.get_history_messages())
        assert self.STUB not in joined
        assert "# Toolkit: browser" in joined
        assert section.transient_entries == []

    def test_the_stub_retires_at_the_next_trim(self):
        section, _ = self._section_after_eviction()
        section.history += self._turns(20, 1100.0)
        joined = self._joined(section.get_history_messages())
        assert self.STUB not in joined
        assert section.transient_entries == []

    def test_no_stub_while_a_newer_row_of_the_toolkit_is_in_the_window(self):
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = ([self._toolkit_row("browser", 1000.0)] + self._turns(20, 1010.0)
                           + [self._toolkit_row("browser", 1040.0, size=10)])
        joined = self._joined(section.get_history_messages())
        assert self.STUB not in joined
        assert section.transient_entries == []
        assert section.loaded_toolkits() == {"browser": 1040.0}

    def test_one_stub_per_evicted_toolkit(self):
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = ([self._toolkit_row("web", 1000.0, size=300), self._toolkit_row("browser", 1002.0)]
                           + self._turns(20, 1010.0))
        joined = self._joined(section.get_history_messages())
        assert 'load_toolkit("web")' in joined and self.STUB in joined
        assert [e["toolkit"] for e in section.transient_entries] == ["web", "browser"]

    def test_the_stub_survives_a_full_prompt_rebuild(self):
        section, _ = self._section_after_eviction()
        with patch.object(section, '_notify_changed'):
            section.clear_transient_entries()
        assert len(section.transient_entries) == 1
        assert self.STUB in self._joined(section.get_history_messages())

    def test_the_prompt_tab_shows_the_stub(self):
        section, _ = self._section_after_eviction()
        assert self.STUB in section.render_for_display()


class TestRemoveFromUserTurns:
    """The generic helper under strip_thinking_instruction: every user turn with string content
    is a candidate, the texts go in the given order, and the count is the number of turns that
    changed — a turn dropped on the way must not hide the one before it from the walk."""

    def test_counts_every_changed_turn_and_removes_the_texts_in_order(self):
        from assistant.prompt.MessageHistory import MessageHistory
        messages = [{"role": "system", "content": "prompt TAG"},
                    {"role": "user", "content": "TAG"},
                    {"role": "user", "content": "\n\nTAG"},
                    {"role": "user", "content": [{"type": "text", "text": "TAG"}]},
                    {"role": "assistant", "content": "TAG"},
                    {"role": "user", "content": "User [Marcus]: hello\n\nTAG"},
                    {"role": "user", "content": "User [Marcus]: bye"}]
        shared_turn = messages[5]
        untouched = [messages[0], messages[3], messages[4], messages[6]]

        assert MessageHistory.remove_from_user_turns(messages, "\n\nTAG", "TAG") == 3

        assert messages == [{"role": "system", "content": "prompt TAG"},
                            {"role": "user", "content": [{"type": "text", "text": "TAG"}]},
                            {"role": "assistant", "content": "TAG"},
                            {"role": "user", "content": "User [Marcus]: hello"},
                            {"role": "user", "content": "User [Marcus]: bye"}]
        assert all(kept is before for kept, before in zip([messages[0], messages[1], messages[2], messages[4]], untouched))
        assert shared_turn["content"] == "User [Marcus]: hello\n\nTAG"  # replaced in the list, not mutated


class TestStripThinkingInstruction:
    """strip_thinking_instruction undoes get_messages(extended_thinking=True) on a built
    message list, wherever append_user_turn put the instruction inside the turn."""

    @pytest.mark.parametrize("before, after", [
        (["(Current time: 14:50)", "INSTRUCTION", "User [Marcus]: hello"], ["(Current time: 14:50)", "User [Marcus]: hello"]),
        (["INSTRUCTION", "User [Marcus]: hello"], ["User [Marcus]: hello"]),
        (["(Current time: 14:50)", "INSTRUCTION"], ["(Current time: 14:50)"]),
    ])
    def test_leaves_what_a_build_without_the_instruction_gives(self, before, after):
        from assistant.prompt.MessageHistory import MessageHistory, THINKING_TAG_INSTRUCTION

        def build(parts):
            messages = [{"role": "assistant", "content": "earlier reply"}]
            for part in parts:
                MessageHistory.append_user_turn(messages, THINKING_TAG_INSTRUCTION if part == "INSTRUCTION" else part)
            return messages

        messages = build(before)
        assert MessageHistory.strip_thinking_instruction(messages) == 1
        assert messages == build(after)

    def test_a_turn_holding_only_the_instruction_is_dropped(self):
        from assistant.prompt.MessageHistory import MessageHistory, THINKING_TAG_INSTRUCTION
        messages = [{"role": "assistant", "content": "earlier reply"}, {"role": "user", "content": THINKING_TAG_INSTRUCTION}]
        assert MessageHistory.strip_thinking_instruction(messages) == 1
        assert messages == [{"role": "assistant", "content": "earlier reply"}]

    def test_nothing_to_strip(self):
        from assistant.prompt.MessageHistory import MessageHistory, THINKING_TAG_INSTRUCTION
        blocks = [{"type": "text", "text": THINKING_TAG_INSTRUCTION}]
        messages = [{"role": "assistant", "content": THINKING_TAG_INSTRUCTION}, {"role": "user", "content": blocks},
                    {"role": "user", "content": "User [Marcus]: hello"}]
        expected = [m.copy() for m in messages]
        assert MessageHistory.strip_thinking_instruction(messages) == 0
        assert messages == expected
