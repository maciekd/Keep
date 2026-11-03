import json
import sqlite3
import time

import pytest
from unittest.mock import patch, Mock

from assistant.prompt.MessageHistory import MessageHistory


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

# ===========================================================================
# TestGetMergedEntries
# ===========================================================================

class TestGetMergedEntries:
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
        section.add_transient_entry("system", "Prompt section `calendar` updated", 1001.0, section="calendar")
        assert calls == [1, 2]

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
# TestTransientEntryPlacement
# ===========================================================================

class TestTransientEntryPlacement:
    def test_a_note_keeps_the_state_its_diff_starts_from(self):
        section = _make_history_section()
        note = section.add_transient_entry("system", "-a\n+b", 1000.0, source="prompt_delta",
                                           section="cars", base="a")
        assert note["base"] == "a"
        rendered = section.get_history_messages()
        assert all("base" not in m for m in rendered) and "a\n" not in rendered[0]["content"].split("-a")[0], \
            "the base is bookkeeping, never a byte of the window"

    def test_first_places_the_row_ahead_of_others_with_the_same_timestamp(self):
        section = _make_history_section()
        section.add_transient_entry("system", "later note", 1000.0, source="prompt_delta", section="cars")
        section.add_transient_entry("system", "bridge", 1000.0, source="prompt_delta", section="cars", first=True)
        rendered = "\n".join(m["content"] for m in section.get_history_messages())
        assert rendered.index("bridge") < rendered.index("later note")

    def test_first_does_not_jump_a_row_it_follows_in_time(self):
        """Only the tie order is at stake: a later timestamp still sorts later."""
        section = _make_history_section()
        section.history = [{"role": "Marcus", "content": "hi", "timestamp": 900.0, "source": None, "location": None}]
        section.add_transient_entry("system", "bridge", 1000.0, first=True)
        rendered = "\n".join(m["content"] for m in section.get_history_messages())
        assert rendered.index("hi") < rendered.index("bridge")


# ===========================================================================
# TestOldestDeltaNote
# ===========================================================================

class TestOldestDeltaNote:
    def test_the_oldest_note_of_the_section_inside_the_window(self):
        section = _make_history_section()
        section.add_transient_entry("system", "cars 2", 1002.0, source="prompt_delta", section="cars", base="b")
        section.add_transient_entry("system", "cars 1", 1001.0, source="prompt_delta", section="cars", base="a")
        section.add_transient_entry("system", "calendar", 1000.0, source="prompt_delta", section="calendar", base="x")
        assert section.oldest_delta_note("cars")["base"] == "a"

    def test_a_note_behind_the_cutoff_does_not_count(self):
        section = _make_history_section()
        section.add_transient_entry("system", "cars 1", 1001.0, source="prompt_delta", section="cars", base="a")
        section.add_transient_entry("system", "cars 2", 1002.0, source="prompt_delta", section="cars", base="b")
        section._trim_cutoff = 1002.0
        assert section.oldest_delta_note("cars")["base"] == "b"

    def test_a_row_that_is_not_a_delta_note_does_not_count(self):
        section = _make_history_section()
        section.add_transient_entry("system", "stub", 1000.0, source="toolkit_evicted", section="cars")
        assert section.oldest_delta_note("cars") is None

    def test_a_tie_goes_to_the_row_that_renders_first(self):
        section = _make_history_section()
        section.add_transient_entry("system", "note", 1000.0, source="prompt_delta", section="cars", base="b")
        section.add_transient_entry("system", "bridge", 1000.0, source="prompt_delta", section="cars", base="a", first=True)
        assert section.oldest_delta_note("cars")["base"] == "a"


# ===========================================================================
# TestWithdrawUnansweredNote
# ===========================================================================

class TestWithdrawUnansweredNote:
    def test_an_unanswered_note_is_withdrawn_and_the_subscribers_hear_of_it(self):
        section = _make_history_section()
        section.history = [{"role": "assistant", "content": "Parked.", "timestamp": 900.0}]
        note = section.add_transient_entry("system", "car note", 1000.0, section="car_info")
        calls = []
        section.subscribe(lambda: calls.append(1))

        assert section.withdraw_unanswered_note(note) is True
        assert section.transient_entries == []
        assert calls == [1]

    def test_a_note_an_assistant_row_follows_stays(self):
        section = _make_history_section()
        note = section.add_transient_entry("system", "car note", 1000.0, section="car_info")
        section.history = [{"role": "Marcus", "content": "Where is the car?", "timestamp": 1001.0},
                           {"role": "assistant", "content": "Driving.", "timestamp": 1002.0}]

        assert section.withdraw_unanswered_note(note) is False
        assert section.transient_entries == [note]

    def test_a_user_row_after_the_note_does_not_seal_it(self):
        """Only a reply ends the trailing user turn a cached request could have ended on."""
        section = _make_history_section()
        note = section.add_transient_entry("system", "car note", 1000.0, section="car_info")
        section.history = [{"role": "Marcus", "content": "Hej", "timestamp": 1001.0}]

        assert section.withdraw_unanswered_note(note) is True

    def test_a_note_no_longer_in_the_list_is_not_withdrawn(self):
        section = _make_history_section()
        note = section.add_transient_entry("system", "car note", 1000.0, section="car_info")
        section.transient_entries = []

        assert section.withdraw_unanswered_note(note) is False

    def test_an_equal_note_that_is_not_the_same_entry_stays(self):
        section = _make_history_section()
        section.add_transient_entry("system", "car note", 1000.0, section="car_info")
        lookalike = dict(section.transient_entries[0])

        assert section.withdraw_unanswered_note(lookalike) is False
        assert len(section.transient_entries) == 1


# ===========================================================================
# TestGetHistoryMessages
# ===========================================================================

class TestGetHistoryMessages:
    """History rendered as standard LLM message items (role/content dicts)."""

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

    def test_a_system_row_renders_whole_past_the_cap(self):
        """The first-run brief is a system row five times the cap, and its setup steps are at
        the end: cut, every later turn lost them (and the assistant enabled Photos before
        iCloud). A user row of the same size is still cut."""
        section = _make_history_section()
        section.max_render_chars = 100
        steps = "\n".join(f"{i}. step {i}" for i in range(1, 9))
        section.history = [
            {"role": "system", "content": "The server has just started. " * 10 + steps, "timestamp": 1000.0,
             "source": "console", "location": None},
            {"role": "assistant", "content": "Hi, who am I talking to?", "timestamp": 1001.0},
            {"role": "Marcus", "content": "p" * 500, "timestamp": 1002.0, "source": None, "location": None},
        ]
        brief, _, user = section.get_history_messages()
        assert brief["content"].startswith("System (") and brief["content"].endswith("8. step 8")
        assert "p" * 100 + "..." in user["content"] and "p" * 101 not in user["content"]

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

    def test_trim_records_the_sections_whose_delta_notes_it_evicted(self):
        """A prompt-delta note is the only thing telling the LLM a section moved on from the
        system prompt's baseline. When the window trims it away, PromptManager has to put a
        fresh one back — so the trim must say which sections lost theirs, and only those."""
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = self._entries(30)
        section.add_transient_entry("system", "Prompt section `calendar` updated:\n-a\n+b",
                                    timestamp=1000.5, source="prompt_delta", section="calendar")
        section.add_transient_entry("system", "Prompt section `cars` updated:\n-x\n+y",
                                    timestamp=1029.5, source="prompt_delta", section="cars")

        joined = "\n".join(m["content"] for m in section.get_history_messages())

        assert "`calendar` updated" not in joined and "`cars` updated" in joined
        assert section.dropped_delta_sections == {"calendar"}

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


class TestFreshRowsDeferTheTrim:
    """Between the high watermark and the ceiling a trim waits while the newest row it would
    drop is younger than trim_fresh_seconds: a burst must not evict the rows the task is
    using. At the ceiling it trims regardless. The other trim tests date their rows in 1970,
    so the gate never opens there."""

    @staticmethod
    def _section(high=2_000, low=1_000, ceiling=4_000, fresh=900):
        section = _make_history_section()
        section.trim_high_bytes = high
        section.trim_low_bytes = low
        section.trim_ceiling_bytes = ceiling
        section.trim_fresh_seconds = fresh
        return section

    @staticmethod
    def _rows(n, age_of_newest, size=100, prefix="m"):
        start = time.time() - age_of_newest - n
        return [{"role": "Marcus", "content": f"{prefix}{i:03d}" + "x" * size, "timestamp": start + i,
                 "source": None, "location": None} for i in range(n)]

    def test_fresh_rows_between_high_and_ceiling_are_kept(self):
        section = self._section()
        section.history = self._rows(30, age_of_newest=60)      # ~3.1KB, all within the last few minutes
        joined = "\n".join(m["content"] for m in section.get_history_messages())
        assert "m000" in joined and section._trim_cutoff == 0.0

    def test_stale_rows_between_high_and_ceiling_are_trimmed_as_before(self):
        section = self._section()
        section.history = self._rows(30, age_of_newest=2_000)   # the whole burst is older than fresh
        joined = "\n".join(m["content"] for m in section.get_history_messages())
        assert "m000" not in joined and section._trim_cutoff > 0.0

    def test_the_ceiling_trims_fresh_rows(self):
        section = self._section()
        section.history = self._rows(45, age_of_newest=60)      # ~4.7KB > ceiling
        msgs = section.get_history_messages()
        joined = "\n".join(m["content"] for m in msgs)
        assert "m000" not in joined and "m044" in joined
        assert len(joined.encode("utf-8")) <= 1_000 + len(msgs) * 2, "a ceiling trim still cuts back to low"

    def test_it_is_the_dropped_rows_age_that_counts_not_the_kept_ones(self):
        """Old rows ahead of a fresh burst: the cut drops only old rows, so it goes ahead even
        though the window is full of fresh ones."""
        section = self._section()
        old = self._rows(20, age_of_newest=3_000)
        fresh = self._rows(8, age_of_newest=30, prefix="f")
        section.history = old + fresh
        joined = "\n".join(m["content"] for m in section.get_history_messages())
        assert "m000" not in joined
        assert all(f"f{i:03d}" in joined for i in range(8)), "every fresh row survives"

    def test_a_deferred_trim_happens_once_the_rows_have_aged(self):
        section = self._section()
        section.history = self._rows(30, age_of_newest=60)
        section.get_history_messages()
        assert section._trim_cutoff == 0.0
        for row in section.history:
            row["timestamp"] -= 1_000
        section.get_history_messages()
        assert section._trim_cutoff > 0.0


# ===========================================================================
# TestBulkyRowsCollapseBeforeTheTrim
# ===========================================================================

class TestBulkyRowsCollapseBeforeTheTrim:
    """Over the high watermark the trim collapses the bulky command rows (a toolkit
    reference, a procedure) to a one-line stub before it drops any row: one command brings a
    reference back, the user's answer of a minute ago nothing does (#135)."""

    @staticmethod
    def _section(high=4_000, low=2_000, ceiling=6_000, collapse=1_000, fresh=900):
        section = _make_history_section()
        section.trim_high_bytes = high
        section.trim_low_bytes = low
        section.trim_ceiling_bytes = ceiling
        section.trim_collapse_bytes = collapse
        section.trim_fresh_seconds = fresh
        return section

    @staticmethod
    def _turns(n, age_of_newest, size=60, prefix="m"):
        start = time.time() - age_of_newest - n
        return [{"role": "Marcus", "content": f"{prefix}{i:03d}" + "x" * size, "timestamp": start + i,
                 "source": None, "location": None} for i in range(n)]

    @staticmethod
    def _command_row(lead, age, size, source=None):
        return {"role": "command", "content": f"result: \n{lead}\n" + "z" * size,
                "timestamp": time.time() - age, "source": source, "location": None}

    @staticmethod
    def _joined(section, window="text"):
        return "\n".join(m["content"] for m in section.get_history_messages(window=window))

    def test_a_stale_bulky_row_collapses_and_the_conversation_stays_whole(self):
        section = self._section()
        turns = self._turns(10, age_of_newest=2_000)
        section.history = turns[:5] + [self._command_row("# Procedure: ha-ui", age=2_010, size=4_000)] + turns[5:]
        joined = self._joined(section)
        assert "zzzz" not in joined and "# Procedure: ha-ui\nzzz" not in joined, "the body is gone"
        assert 'began "# Procedure: ha-ui"' in joined and "3.9KB" in joined, "the stub says what stood there"
        assert all(f"m{i:03d}" in joined for i in range(10)), "every conversation row survives"
        assert section._trim_cutoff == 0.0, "nothing was dropped"

    def test_at_the_ceiling_the_fresh_bulky_row_goes_not_the_fresh_exchange(self):
        """The #135 shape: a decision 30 s old, then a huge procedure result that pushes the
        window past the ceiling. The procedure collapses; the decision stays."""
        section = self._section()
        exchange = self._turns(3, age_of_newest=30, prefix="d")
        section.history = (self._turns(20, age_of_newest=2_000) + exchange
                           + [self._command_row("# Procedure: ha-ui", age=20, size=6_000)])
        joined = self._joined(section)
        assert all(f"d{i:03d}" in joined for i in range(3))
        assert "zzzz" not in joined

    def test_under_the_ceiling_a_fresh_bulky_row_stays_whole(self):
        section = self._section()
        section.history = self._turns(20, age_of_newest=2_000) + [self._command_row("# Procedure: p", age=20, size=3_000)]
        joined = self._joined(section)
        assert "zzzz" in joined, "the task is still using it"
        assert section._trim_cutoff > 0.0, "the stale turns ahead of it went instead"

    def test_the_collapse_stops_at_the_low_watermark_newest_rows_last(self):
        section = self._section(low=2_500)
        section.history = (self._turns(4, age_of_newest=3_000)
                           + [self._command_row("# Procedure: old", age=2_500, size=3_000)]
                           + self._turns(4, age_of_newest=2_000, prefix="n")
                           + [self._command_row("# Procedure: new", age=1_500, size=1_500)])
        joined = self._joined(section)
        assert 'began "# Procedure: old"' in joined and "# Procedure: old\nzzz" not in joined
        assert "# Procedure: new\nzzz" in joined, "the newer one was not needed to reach low"

    def test_a_collapse_is_sticky_so_the_window_holds_still(self):
        section = self._section()
        section.history = self._turns(6, age_of_newest=2_000) + [self._command_row("# Procedure: p", age=1_990, size=4_000)]
        first = section.get_history_messages()
        section.history.append({"role": "Marcus", "content": "and now?", "timestamp": time.time(),
                                "source": None, "location": None})
        second = section.get_history_messages()
        assert "zzzz" not in second[0]["content"]
        assert second[0]["content"].startswith(first[0]["content"]), "the earlier window is a byte prefix"

    def test_a_collapsed_toolkit_reference_is_evicted_not_loaded(self):
        """The row is in the window but its text is gone: the eviction stub names the
        toolkit, loaded_toolkits leaves it out (the preloader sends it again), and a delta
        note of its carried section is not rendered against data the LLM no longer holds."""
        section = self._section()
        reference = self._command_row("# Toolkit: home", age=2_010, size=4_000)
        section.history = self._turns(10, age_of_newest=2_000) + [reference]
        section.add_transient_entry("system", "home moved on", time.time() - 1_000, source="prompt_delta", section="home")
        msgs = section.get_history_messages()
        joined = "\n".join(m["content"] for m in msgs)
        assert "# Toolkit: home\nzzz" not in joined and "zzzz" not in joined
        assert msgs[0]["content"].startswith("System (") and "home" in msgs[0]["content"].split("\n")[0]
        assert section.loaded_toolkits() == {}
        assert "home moved on" not in joined

    def test_a_small_command_row_is_never_collapsed(self):
        section = self._section()
        section.history = self._turns(30, age_of_newest=2_000) + [self._command_row("result row", age=1_990, size=200)]
        joined = self._joined(section)
        assert "result row\nzzz" in joined


# ===========================================================================
# TestLoadSaveHistory
# ===========================================================================

class TestLoadSaveHistory:
    def test_load_db_error_returns_empty(self):
        section = _make_history_section()
        with patch('assistant.prompt.MessageHistory.FloxyDatabase') as MockDB:
            MockDB.get_instance.side_effect = Exception("DB error")
            section._load_history()
        assert section.history == []

# ===========================================================================
# TestRenderForDisplay
# ===========================================================================

class TestRenderForDisplay:
    """render_for_display() powers the console PromptTab viewer."""

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

    def test_shows_the_bytes_the_llm_gets_so_a_long_turn_shows_cut(self):
        section = _make_history_section()
        section.max_render_chars = 50
        long_text = "z" * 500
        section.history = [{"role": "user", "content": long_text, "timestamp": 1000.0}]
        out = section.render_for_display()
        sent = section.get_history_messages()[0]["content"]
        assert long_text not in out
        assert sent in out

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

    def test_a_note_never_leaves_the_window_short_of_a_trim(self):
        """The window is append-only: removing a note would change the rendered bytes from
        that row on and lose the prefix cache the next request reads. A newer note for the
        same section goes after the old one; both stay."""
        section = _make_history_section()
        section.add_transient_entry("system", "weather diff", 1000.0, section="weather")
        section.add_transient_entry("system", "calendar diff", 1001.0, section="calendar")
        before = section.get_history_messages()
        section.add_transient_entry("system", "weather diff 2", 1002.0, section="weather")
        after = section.get_history_messages()
        assert after[0]["content"].startswith(before[0]["content"]), "the earlier bytes are a prefix of the later"
        assert [e["section"] for e in section.transient_entries] == ["weather", "calendar", "weather"]
        assert not hasattr(section, "clear_transient_entries")
        assert not hasattr(section, "clear_transient_entries_for_section")

    def test_a_trim_forgets_the_notes_it_left_behind_the_cutoff(self):
        """Pruning is bookkeeping, not a change: the rows behind the cutoff are not rendered
        anyway, so the window stays byte-identical, while notes at or after the cutoff stay."""
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = [{"role": "Marcus", "content": f"m{i:03d}" + "x" * 100, "timestamp": 1000.0 + i,
                            "source": None, "location": None} for i in range(30)]
        section.add_transient_entry("system", "early note", 1000.5, source="prompt_delta", section="calendar")
        section.add_transient_entry("system", "late note", 1029.5, source="prompt_delta", section="calendar")
        rendered = section.get_history_messages()

        assert section._trim_cutoff > 1000.5, "the trim moved past the early note"
        assert [e["content"] for e in section.transient_entries] == ["late note"]
        assert "early note" not in "\n".join(m["content"] for m in rendered)
        assert "late note" in "\n".join(m["content"] for m in rendered)
        assert section.get_history_messages() == rendered

    def test_pruning_keeps_the_eviction_stub_behind_the_cutoff(self):
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = [{"role": "command", "content": "result: \n# Toolkit: browser\n" + "z" * 800,
                            "timestamp": 1000.0, "source": None, "location": None}]
        section.history += [{"role": "Marcus", "content": f"m{i:03d}" + "x" * 100, "timestamp": 1010.0 + i,
                             "source": None, "location": None} for i in range(20)]
        section.add_transient_entry("system", "old note", 1001.0, source="prompt_delta", section="weather")
        section.get_history_messages()
        sources = [e["source"] for e in section.transient_entries]
        assert sources == [section.SOURCE_TOOLKIT_EVICTED]
        assert section.transient_entries[0]["timestamp"] < section._trim_cutoff


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

    def test_a_preloaded_row_is_in_its_own_window_only(self):
        """A toolkit voice preloaded (its row stamped preload:voice) is in voice's window and
        out of a chat's; a toolkit the LLM loaded itself (no stamp) is in both."""
        section = self._section_with([
            {"role": "Marcus", "content": "lights off", "timestamp": 100.0, "source": "VoiceToVoice"},
            {"role": "command", "content": "result: \n# Toolkit: home\nhome text", "timestamp": 101.0,
             "source": MessageHistory.preload_source("voice")},
            {"role": "command", "content": "result: \n# Toolkit: web\nweb text", "timestamp": 102.0},
            {"role": "assistant", "content": "Done", "timestamp": 103.0},
        ])
        voice = "\n".join(m["content"] for m in section.get_history_messages(window="voice"))
        text = "\n".join(m["content"] for m in section.get_history_messages(window="text"))
        assert "# Toolkit: home" in voice and "# Toolkit: web" in voice
        assert "# Toolkit: home" not in text and "# Toolkit: web" in text
        assert "lights off" in text and "Done" in text, "the conversation itself is every window's"
        assert set(section.loaded_toolkits("voice")) == {"home", "web"}
        assert set(section.loaded_toolkits("text")) == {"web"}
        assert set(section.loaded_toolkits(None)) == {"home", "web"}, "any window: the delta tracking's view"

    def test_a_carried_delta_note_follows_its_toolkits_reference(self):
        """Home's delta notes diff against the data the last load sent: a window without
        that load (chat, after a voice preload of home) must not get them, and gets the
        ones written after it loaded home itself. A note of a prompt section (no toolkit
        carries it) is every window's."""
        section = self._section_with([
            {"role": "command", "content": "result: \n# Toolkit: home\nhome text", "timestamp": 100.0,
             "source": MessageHistory.preload_source("voice")},
        ])
        section.add_transient_entry("system", "HOME AUTOMATION DATA updated:\n+lamp on", 101.0,
                                    source=MessageHistory.SOURCE_PROMPT_DELTA, section="home")
        section.add_transient_entry("system", "Prompt section `calendar` updated:\n+lunch", 102.0,
                                    source=MessageHistory.SOURCE_PROMPT_DELTA, section="calendar")
        text = "\n".join(m["content"] for m in section.get_history_messages(window="text"))
        assert "+lamp on" not in text and "+lunch" in text
        voice = "\n".join(m["content"] for m in section.get_history_messages(window="voice"))
        assert "+lamp on" in voice and "+lunch" in voice

        section.history.append({"role": "command", "content": "result: \n# Toolkit: home\nhome text",
                                "timestamp": 103.0})
        section.add_transient_entry("system", "HOME AUTOMATION DATA updated:\n+lamp off", 104.0,
                                    source=MessageHistory.SOURCE_PROMPT_DELTA, section="home")
        text = "\n".join(m["content"] for m in section.get_history_messages(window="text"))
        assert "+lamp on" not in text and "+lamp off" in text

    def test_the_display_shows_one_window_and_marks_the_preloaded(self):
        section = self._section_with([
            {"role": "command", "content": "result: \n# Toolkit: home\nhome text", "timestamp": 100.0,
             "source": MessageHistory.preload_source("voice")},
            {"role": "command", "content": "result: \n# Toolkit: web\nweb text", "timestamp": 101.0},
        ])
        shown = section.render_for_display("voice")
        assert "home text" in shown and "web text" in shown
        headers = [line for line in shown.splitlines() if line.startswith("### Command result")]
        assert len(headers) == 2
        assert "voice window only" in headers[0] and "window only" not in headers[1]
        assert "home text" not in section.render_for_display("text")

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
# TestToolkitEvictionStubs — a trim that drops toolkit rows leaves one stub naming them
# ===========================================================================

class TestToolkitEvictionStubs:
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

    @staticmethod
    def _stubs(section):
        return [e for e in section.transient_entries if e["source"] == section.SOURCE_TOOLKIT_EVICTED]

    def test_the_stub_stands_at_the_head_and_names_the_toolkit(self):
        section, msgs = self._section_after_eviction()
        joined = self._joined(msgs)
        assert "# Toolkit: browser" not in joined, "the reference itself is gone"
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"].startswith("System (")
        assert "browser" in msgs[0]["content"].split("\n")[0]
        assert [e["toolkits"] for e in self._stubs(section)] == [["browser"]]
        assert section._trim_cutoff > 1002.0, "the stub is exempt from the cutoff, not the cutoff"

    def test_the_stub_is_not_a_loaded_toolkit(self):
        section, _ = self._section_after_eviction()
        assert section.loaded_toolkits() == {}

    def test_rendering_again_keeps_the_same_stub(self):
        section, msgs = self._section_after_eviction()
        stub = self._stubs(section)[0]
        again = section.get_history_messages()
        assert again == msgs, "the head must stay byte-stable between renders for the cache"
        assert self._stubs(section) == [stub] and self._stubs(section)[0] is stub

    def test_the_stub_holds_still_when_the_toolkit_is_loaded_again(self):
        """The stub is the window's first row: a byte of it changing is a cache miss on the
        whole window, so a reload leaves it as it is — the reload row below is what tells
        the LLM the toolkit is current again."""
        section, before = self._section_after_eviction()
        stub = self._stubs(section)[0]
        section.history.append(self._toolkit_row("browser", 2000.0, size=10))
        msgs = section.get_history_messages()
        assert msgs[0]["content"].startswith(before[0]["content"]), "the earlier window is a byte prefix of the new"
        assert msgs[0]["content"].split("\n")[0] == before[0]["content"].split("\n")[0]
        assert "# Toolkit: browser" in self._joined(msgs)
        assert self._stubs(section) == [stub] and self._stubs(section)[0] is stub
        assert section.loaded_toolkits() == {"browser": 2000.0}

    def test_the_stub_retires_at_the_next_trim(self):
        section, _ = self._section_after_eviction()
        section.history += self._turns(20, 1100.0)
        msgs = section.get_history_messages()
        assert not msgs[0]["content"].startswith("System (")
        assert self._stubs(section) == []

    def test_the_next_trims_stub_names_only_what_that_trim_evicted(self):
        """A toolkit not loaded again within one window's worth of conversation is not in use:
        the second trim's stub names the toolkit it evicted and the first trim's name is gone."""
        section, _ = self._section_after_eviction()
        section.history += [self._toolkit_row("web", 1100.0)] + self._turns(20, 1110.0)
        msgs = section.get_history_messages()
        head = msgs[0]["content"].split("\n")[0]
        assert head.startswith("System (") and "web" in head and "browser" not in head
        assert [(e["toolkits"], e["timestamp"]) for e in self._stubs(section)] == [(["web"], 1100.0)]
        assert section.loaded_toolkits() == {}

    def test_no_stub_while_a_newer_row_of_the_toolkit_is_in_the_window(self):
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = ([self._toolkit_row("browser", 1000.0)] + self._turns(20, 1010.0)
                           + [self._toolkit_row("browser", 1040.0, size=10)])
        msgs = section.get_history_messages()
        assert not msgs[0]["content"].startswith("System (")
        assert self._stubs(section) == []
        assert section.loaded_toolkits() == {"browser": 1040.0}

    def _section_after_evicting_two(self):
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = ([self._toolkit_row("web", 1000.0, size=300), self._toolkit_row("browser", 1002.0)]
                           + self._turns(20, 1010.0))
        return section, section.get_history_messages()

    def test_one_stub_names_every_evicted_toolkit(self):
        section, msgs = self._section_after_evicting_two()
        head = msgs[0]["content"].split("\n")[0]
        assert "web" in head and "browser" in head
        assert msgs[0]["content"].count("System (") == 1
        assert [e["toolkits"] for e in self._stubs(section)] == [["browser", "web"]]
        assert self._stubs(section)[0]["timestamp"] == 1002.0, "the newest evicted row's place"

    def test_a_preloaded_row_puts_no_name_in_the_stub(self):
        """The preloader sends its toolkit again on the next request of the window, so a stub
        telling the model to load it again would stand over the reload itself."""
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        preloaded = dict(self._toolkit_row("setup", 1002.0), source=section.preload_source("text"))
        section.history = [preloaded] + self._turns(20, 1010.0)
        msgs = section.get_history_messages()
        assert "# Toolkit: setup" not in self._joined(msgs), "the row itself is trimmed"
        assert not msgs[0]["content"].startswith("System (")
        assert self._stubs(section) == []

    def test_a_preloaded_row_beside_a_loaded_one_leaves_only_the_loaded_name(self):
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        preloaded = dict(self._toolkit_row("setup", 1000.0, size=300), source=section.preload_source("text"))
        section.history = [preloaded, self._toolkit_row("browser", 1002.0)] + self._turns(20, 1010.0)
        msgs = section.get_history_messages()
        head = msgs[0]["content"].split("\n")[0]
        assert "browser" in head and "setup" not in head
        assert [e["toolkits"] for e in self._stubs(section)] == [["browser"]]

    def test_a_reloaded_toolkit_does_not_change_the_stub_naming_two(self):
        section, before = self._section_after_evicting_two()
        section.history.append(self._toolkit_row("web", 2000.0, size=10))
        msgs = section.get_history_messages()
        assert msgs[0]["content"].startswith(before[0]["content"])
        assert [e["toolkits"] for e in self._stubs(section)] == [["browser", "web"]]
        assert self._stubs(section)[0]["timestamp"] == 1002.0

    def test_the_stub_survives_a_full_prompt_rebuild(self):
        """A rebuild re-bakes the sections into the prompt; it touches no row of the window,
        the stub included."""
        section, _ = self._section_after_eviction()
        section.add_transient_entry("system", "weather diff", 1030.0, source="prompt_delta", section="weather")
        head_before = section.get_history_messages()[0]
        from assistant.prompt.PromptManager import PromptManager
        manager = PromptManager(test_suite=True)
        manager.message_history = section
        manager._assemble_prompt(force=True)
        assert [e["toolkits"] for e in self._stubs(section)] == [["browser"]]
        assert section.get_history_messages()[0] == head_before
        assert [e["content"] for e in section.transient_entries if e["source"] == "prompt_delta"] == ["weather diff"]

    def test_the_prompt_tab_shows_the_stub(self):
        section, _ = self._section_after_eviction()
        under_header = section.render_for_display().split("### System (", 1)[1].split("\n### ", 1)[0]
        assert "browser" in under_header


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


# ===========================================================================
# TestBytePrefixExchange
# ===========================================================================

class TestBytePrefixExchange:
    """A byte-prefix request (Ollama's MLX runner) sends its trailing turn exactly as the
    history renders it afterwards: the clock note first, then the timestamped query. The
    persisted exchange must re-render those same bytes or the runner's restore point is lost."""

    _note = "(Current time: 14:40, Saturday, 2025-03-15)"

    def _section_writing_to_memory(self):
        """A history whose rows land in memory the way the database stores them (role lowercased)."""
        section = _make_history_section()
        rows = []

        def store(role, content, timestamp, source=None, location=None):
            rows.append({"role": role.lower(), "content": content, "timestamp": timestamp,
                         "source": source, "location": location})

        patch.object(section, '_save_history_entry', side_effect=store).start()
        patch.object(section, '_load_history',
                     side_effect=lambda: setattr(section, 'history', sorted(rows, key=lambda r: r["timestamp"]))).start()
        patch.object(section, '_notify_changed').start()
        return section, rows

    def test_the_persisted_exchange_re_renders_the_request_turn_byte_for_byte(self):
        from datetime import datetime
        from assistant.prompt.MessageHistory import MessageHistory
        from assistant.tools.MessageFormatter import MessageFormatter
        sent_at = datetime(2025, 3, 15, 14, 40, 0).timestamp()
        # What RequestProcessor + DialogManager sent: the note appended to the tail, the query
        # folded onto it (the client spelled the speaker "Marcus").
        request_tail: list[dict] = []
        MessageHistory.append_user_turn(request_tail, self._note)
        MessageHistory.append_user_turn(
            request_tail, MessageFormatter.format_message("Marcus", sent_at, "Console", None, "what's on today?"))
        section, _ = self._section_writing_to_memory()
        try:
            section.add_exchange("Marcus", "what's on today?", "Nothing booked.", "Console", None,
                                 timestamp=sent_at, note=self._note)
            rendered = section.get_history_messages()
        finally:
            patch.stopall()

        assert [m["role"] for m in rendered] == ["user", "assistant"]
        assert rendered[0]["content"] == request_tail[0]["content"]
        assert rendered[1]["content"] == "Nothing booked."

    def test_the_note_row_precedes_the_user_row_and_both_are_dated_at_the_request(self):
        sent_at = 1_700_000_000.0
        section, rows = self._section_writing_to_memory()
        try:
            with patch('assistant.prompt.MessageHistory.time.time', return_value=sent_at + 42.0):
                section.add_exchange("Marcus", "what's on today?", "Nothing booked.", "Console", None,
                                     timestamp=sent_at, note=self._note)
        finally:
            patch.stopall()

        by_role = {r["role"]: r for r in rows}
        assert by_role["marcus"]["timestamp"] == sent_at, "the header sent carried the request time, so must the row"
        assert by_role["note"]["timestamp"] < by_role["marcus"]["timestamp"] < by_role["assistant"]["timestamp"]
        assert by_role["note"]["content"] == self._note

    def test_a_request_time_behind_the_trim_cutoff_is_dated_now_so_the_row_stays_in_the_window(self):
        sent_at = 1_700_000_000.0
        section, rows = self._section_writing_to_memory()
        section._trim_cutoff = sent_at + 10.0          # the window head moved on while the request ran
        try:
            with patch('assistant.prompt.MessageHistory.time.time', return_value=sent_at + 42.0):
                section.add_exchange("Marcus", "what's on today?", "Nothing booked.", "Console", None,
                                     timestamp=sent_at, note=self._note)
            rendered = section.get_history_messages()
        finally:
            patch.stopall()

        assert all(r["timestamp"] >= section._trim_cutoff for r in rows)
        assert any("what's on today?" in m["content"] for m in rendered)
        assert any(self._note in m["content"] for m in rendered), "the note must move with the row it precedes"

    def test_a_classic_exchange_stores_no_note_row(self):
        section, rows = self._section_writing_to_memory()
        try:
            section.add_exchange("Marcus", "what's on today?", "Nothing booked.", "Console", None)
        finally:
            patch.stopall()
        assert {r["role"] for r in rows} == {"marcus", "assistant"}


@pytest.mark.unit
class TestARenderThatChangesNothingLogsNothing:
    """The TUI renders the window many times a second and nothing changes between renders,
    so the trim's lines fire once per state: on the production server the eviction, the
    "stays whole" and the "nothing to collapse" lines had made up tens of thousands of lines
    of the log, dozens within one second."""

    @staticmethod
    def _turns(n, start_ts, size=100):
        return [{"role": "Marcus", "content": f"m{i:03d}" + "x" * size, "timestamp": start_ts + i,
                 "source": None, "location": None} for i in range(n)]

    @staticmethod
    def _command_row(lead, timestamp, size):
        return {"role": "command", "content": f"result: \n{lead}\n" + "z" * size,
                "timestamp": timestamp, "source": None, "location": None}

    @staticmethod
    def _logged(section, renders=3):
        """The info and trace calls of each of `renders` renders of an unchanged window."""
        counts = []
        with patch("assistant.prompt.MessageHistory.info") as info, patch("assistant.prompt.MessageHistory.trace") as trace:
            for _ in range(renders):
                section.get_history_messages()
                counts.append((info.call_count, trace.call_count))
        return [(i - pi, t - pt) for (i, t), (pi, pt) in zip(counts, [(0, 0)] + counts[:-1])]

    def test_an_evicted_toolkit_is_logged_at_the_trim_not_on_every_render(self):
        section = _make_history_section()
        section.trim_high_bytes = 2_000
        section.trim_low_bytes = 1_000
        section.history = [
            {"role": "Marcus", "content": "open ellos", "timestamp": 1000.0, "source": None, "location": None},
            {"role": "assistant", "content": 'load_toolkit("browser")', "timestamp": 1001.0},
            self._command_row("# Toolkit: browser", 1002.0, 800),
        ] + self._turns(20, 1010.0)
        first, *later = self._logged(section)
        assert first[0] >= 1, "the trim and the eviction are logged when they happen"
        assert later == [(0, 0), (0, 0)]
        assert len([e for e in section.transient_entries if e["source"] == section.SOURCE_TOOLKIT_EVICTED]) == 1

    def test_a_bulky_row_too_young_to_collapse_is_reported_once_per_window_size(self):
        """Over the high watermark with a fresh bulky row under the ceiling, the trim waits;
        the two lines saying so (the row stays whole, the trim is deferred) fire for the
        window size they describe and not again while it holds still."""
        section = _make_history_section()
        section.trim_high_bytes = 4_000
        section.trim_low_bytes = 2_000
        section.trim_ceiling_bytes = 8_000
        section.trim_collapse_bytes = 1_000
        section.trim_fresh_seconds = 900
        now = time.time()
        section.history = self._turns(8, now - 400) + [self._command_row("# Procedure: ha-ui", now - 300, 4_000)] \
            + self._turns(8, now - 200)
        first, *later = self._logged(section)
        assert first[1] >= 1, "the deferral is traced once"
        assert later == [(0, 0), (0, 0)]
        assert section._collapsed == set(), "the fresh row stayed whole"
