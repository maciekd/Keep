"""The app chat's item log: sequence numbers, replay from a cursor, the cap, listeners, the file."""
import json
import threading
from types import SimpleNamespace

import pytest

from interfaces.app import Transcript as transcript_module
from interfaces.app.Transcript import Subscription, Transcript


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def transcript(tmp_path, clock):
    return Transcript(tmp_path, autosave=False, clock=clock)


def replay(transcript, after=0, log=None):
    lines, subscription = transcript.subscribe(after, transcript.log if log is None else log, lambda: None)
    transcript.unsubscribe(subscription)
    return lines[1:-1]    # without the line naming the log and the one that ends the catching up


def ids(lines):
    return [line["item"]["id"] if "item" in line else ("removed", line["remove"]) for line in lines]


class TestItems:
    def test_a_changed_item_keeps_its_place_and_its_time(self, transcript, clock):
        transcript.upsert("a", "command", text="Checking", state="running")
        transcript.upsert("b", "assistant", html="Hi", final=False)
        clock.now += 30
        transcript.upsert("a", "command", text="Checking", state="done")
        first, second = transcript.items()
        assert (first["id"], first["state"], second["id"]) == ("a", "done", "b")
        assert first["order"] < second["order"]
        assert first["at"] == second["at"]

    def test_a_change_replaces_the_fields(self, transcript):
        transcript.upsert("a", "approval", token="t", outcome="")
        transcript.upsert("a", "approval", token="t")
        assert "outcome" not in transcript.item("a")


class TestReplay:
    def test_a_fresh_app_gets_every_item(self, transcript):
        transcript.upsert("a", "user", text="hi")
        transcript.upsert("b", "assistant", html="Hello", final=True)
        assert ids(replay(transcript)) == ["a", "b"]

    def test_from_a_cursor_only_what_changed_since(self, transcript):
        transcript.upsert("a", "user", text="hi")
        transcript.upsert("b", "assistant", html="Hel", final=False)
        cursor = replay(transcript)[-1]["seq"]
        transcript.upsert("b", "assistant", html="Hello", final=True)
        transcript.upsert("c", "notice", html="By the way")
        lines = replay(transcript, after=cursor)
        assert ids(lines) == ["b", "c"]
        assert lines[0]["item"]["html"] == "Hello"
        assert all(line["seq"] > cursor for line in lines)

    def test_a_removed_item_is_told_to_whoever_saw_it(self, transcript):
        transcript.upsert("a", "assistant", html="Hm", final=False)
        cursor = replay(transcript)[-1]["seq"]
        transcript.remove("a")
        assert ids(replay(transcript, after=cursor)) == [("removed", "a")]
        assert replay(transcript) == []

    def test_a_cursor_from_another_log_starts_over(self, transcript):
        transcript.upsert("a", "user", text="hi")
        lines = replay(transcript, after=1, log="some-other-log")
        assert lines[0] == {"reset": True}
        assert ids(lines[1:]) == ["a"]

    def test_a_cursor_ahead_of_the_log_starts_over(self, transcript):
        transcript.upsert("a", "user", text="hi")
        assert replay(transcript, after=99)[0] == {"reset": True}

    def test_a_cursor_older_than_the_log_reaches_starts_over(self, transcript, monkeypatch):
        monkeypatch.setattr(transcript_module, "CAP_ITEMS", 3)
        transcript.upsert("a", "user", text="1")
        cursor = replay(transcript)[-1]["seq"]
        for name in "bcde":
            transcript.upsert(name, "user", text=name)
        lines = replay(transcript, after=cursor)
        assert lines[0] == {"reset": True}
        assert ids(lines[1:]) == ["c", "d", "e"]

    def test_a_cursor_older_than_the_removals_still_remembered_starts_over(self, transcript, monkeypatch):
        """A removal the log has forgotten cannot be told: the app would keep the item for ever."""
        monkeypatch.setattr(transcript_module, "MAX_TOMBSTONES", 2)
        for name in "abcd":
            transcript.upsert(name, "user", text=name)
        saw_them_all = replay(transcript)[-1]["seq"]
        transcript.remove("a")
        saw_the_first_removal = saw_them_all + 1
        transcript.remove("b")
        transcript.remove("c")
        lines = replay(transcript, after=saw_them_all)
        assert lines[0] == {"reset": True} and ids(lines[1:]) == ["d"]
        assert ids(replay(transcript, after=saw_the_first_removal)) == [("removed", "b"), ("removed", "c")]

    def test_an_item_that_comes_back_under_its_id_is_not_told_as_removed(self, transcript):
        transcript.upsert("a", "user", text="hi")
        cursor = replay(transcript)[-1]["seq"]
        transcript.remove("a")
        transcript.upsert("a", "user", text="hi again")
        lines = replay(transcript, after=cursor)
        assert ids(lines) == ["a"] and lines[0]["item"]["text"] == "hi again"

    def test_a_current_cursor_gets_nothing(self, transcript):
        transcript.upsert("a", "user", text="hi")
        assert replay(transcript, after=replay(transcript)[-1]["seq"]) == []


class TestCap:
    def test_items_past_their_age_go_when_a_new_one_arrives(self, transcript, clock):
        transcript.upsert("old", "user", text="long ago")
        clock.now += transcript_module.MAX_AGE_SECONDS + 1
        transcript.upsert("new", "user", text="now")
        assert [item["id"] for item in transcript.items()] == ["new"]

    def test_a_dropped_photo_takes_its_file_along(self, transcript, clock, tmp_path):
        url = transcript.add_media(b"jpeg-bytes", "jpg")
        transcript.upsert("photo", "media", url=url, poster="")
        media_id = url.rsplit("/", 1)[1]
        assert transcript.media(media_id) == (b"jpeg-bytes", "image/jpeg")
        clock.now += transcript_module.MAX_AGE_SECONDS + 1
        transcript.upsert("new", "user", text="now")
        assert transcript.media(media_id) is None

    def test_a_dropped_message_takes_its_pictures_along(self, transcript, clock):
        url = transcript.add_media(b"jpeg-bytes", "jpg")
        transcript.upsert("sent", "user", text="look", images=[{"url": url, "width": 4, "height": 3}])
        media_id = url.rsplit("/", 1)[1]
        clock.now += transcript_module.MAX_AGE_SECONDS + 1
        transcript.upsert("new", "user", text="now", images=[])
        assert transcript.media(media_id) is None

    def test_a_dropped_command_takes_the_file_of_its_result_along(self, transcript, clock):
        url = transcript.add_media("Battery 78 %".encode(), "txt")
        transcript.upsert("ran", "command", text="Checking the car", state="done", detail="done", result=url, images=[])
        media_id = url.rsplit("/", 1)[1]
        assert transcript.media(media_id)[0] == b"Battery 78 %"
        clock.now += transcript_module.MAX_AGE_SECONDS + 1
        transcript.upsert("new", "user", text="now")
        assert transcript.media(media_id) is None

    def test_a_command_without_a_result_drops_cleanly(self, transcript, clock):
        transcript.upsert("ran", "command", text="Turning on the kitchen", state="done", detail="done", result="", images=[])
        transcript.upsert("old", "command", text="Checking the car", state="running", detail="")
        clock.now += transcript_module.MAX_AGE_SECONDS + 1
        transcript.upsert("new", "user", text="now")
        assert [item["id"] for item in transcript.items()] == ["new"]


class TestMedia:
    @pytest.mark.parametrize("media_id", ["../app_chat.json", "..%2Fconfig.yaml", "", "abc.jpg", "/etc/passwd"])
    def test_an_id_that_is_not_one_reads_no_file(self, transcript, media_id):
        transcript.upsert("a", "user", text="hi")
        transcript.flush()
        assert transcript.media(media_id) is None


class TestListeners:
    def test_a_listener_is_woken_with_each_change_and_the_working_line(self, transcript):
        woken = []
        _, subscription = transcript.subscribe(0, "", lambda: woken.append(1))
        transcript.upsert("a", "user", text="hi")
        transcript.set_working({"phase": "waiting"})
        lines = subscription.drain()
        assert [sorted(line) for line in lines] == [["item", "seq"], ["working"]]
        assert len(woken) == 2
        assert subscription.drain() == []

    def test_in_a_burst_the_newest_state_of_an_item_wins(self, transcript):
        _, subscription = transcript.subscribe(0, "", lambda: None)
        for text in ("H", "He", "Hel"):
            transcript.upsert("reply", "assistant", html=text, final=False)
        transcript.upsert("other", "notice", html="x")
        transcript.upsert("reply", "assistant", html="Hello", final=True)
        lines = subscription.drain()
        assert ids(lines) == ["other", "reply"]
        assert lines[1]["item"]["html"] == "Hello"

    def test_a_removal_supersedes_the_changes_waiting_before_it(self, transcript):
        _, subscription = transcript.subscribe(0, "", lambda: None)
        transcript.upsert("reply", "assistant", html="H", final=False)
        transcript.remove("reply")
        assert ids(subscription.drain()) == [("removed", "reply")]

    def test_the_working_line_is_not_logged_but_a_joining_listener_sees_the_current_one(self, transcript):
        transcript.set_working({"phase": "thinking"})
        lines, subscription = transcript.subscribe(0, "", lambda: None)
        assert lines[-2:] == [{"working": {"phase": "thinking"}}, {"live": True}]
        transcript.unsubscribe(subscription)
        transcript.set_working(None)
        assert replay(transcript) == []

    def test_an_unchanged_working_line_wakes_nobody(self, transcript):
        woken = []
        transcript.subscribe(0, "", lambda: woken.append(1))
        transcript.set_working(None)
        transcript.set_working({"phase": "waiting"})
        transcript.set_working({"phase": "waiting"})
        assert len(woken) == 1

    def test_listeners_are_counted_while_they_listen(self, transcript):
        _, subscription = transcript.subscribe(0, "", lambda: None)
        assert transcript.listeners() == 1
        transcript.unsubscribe(subscription)
        assert transcript.listeners() == 0


class TestStopping:
    def test_a_stopping_server_ends_every_listener_and_wakes_it(self, transcript):
        woken = []
        _, subscription = transcript.subscribe(0, "", lambda: woken.append(1))
        assert not subscription.ended
        transcript.end_listeners()
        assert subscription.ended and woken

    def test_a_listener_that_joins_a_stopping_server_is_ended_at_once(self, transcript):
        transcript.end_listeners()
        _, subscription = transcript.subscribe(0, "", lambda: None)
        assert subscription.ended


class TestFile:
    def test_a_restarted_server_continues_the_same_log(self, transcript, tmp_path, clock):
        transcript.upsert("a", "user", text="hi")
        transcript.upsert("b", "assistant", html="Hello", final=True)
        cursor = replay(transcript)[-1]["seq"]
        transcript.remove("a")
        transcript.flush(closed=True)
        again = Transcript(tmp_path, autosave=False, clock=clock)
        assert again.log == transcript.log
        assert ids(replay(again, after=cursor)) == [("removed", "a")]
        again.upsert("c", "user", text="more")
        assert [line["seq"] for line in replay(again, after=cursor)] == [cursor + 1, cursor + 2]

    def test_a_run_that_was_killed_after_its_last_save_is_not_continued_under_its_numbers(self, transcript, tmp_path, clock):
        transcript.upsert("a", "user", text="hi")
        transcript.flush()
        transcript.upsert("b", "assistant", html="Hello", final=True)     # never saved: the run is killed
        cursor, log = replay(transcript)[-1]["seq"], transcript.log
        again = Transcript(tmp_path, autosave=False, clock=clock)
        again.upsert("c", "user", text="more")
        lines = replay(again, after=cursor, log=log)
        assert lines[0] == {"reset": True} and ids(lines[1:]) == ["a", "c"]

    def test_a_run_killed_before_its_first_save_does_not_pass_for_the_closed_one_before_it(self, transcript, tmp_path, clock):
        transcript.upsert("a", "user", text="hi")
        transcript.flush(closed=True)
        killed = Transcript(tmp_path, autosave=False, clock=clock)
        killed.upsert("b", "user", text="seen by a phone, never saved")
        cursor = replay(killed)[-1]["seq"]
        again = Transcript(tmp_path, autosave=False, clock=clock)
        again.upsert("c", "user", text="more")
        lines = replay(again, after=cursor, log=killed.log)
        assert lines[0] == {"reset": True} and ids(lines[1:]) == ["a", "c"]

    def test_a_change_after_the_closing_save_is_not_continued_under_its_numbers(self, transcript, tmp_path, clock):
        transcript.upsert("a", "user", text="hi")
        transcript.flush(closed=True)
        transcript.upsert("b", "assistant", html="Stopped.", final=True)    # a cancelled reply settling late
        transcript.upsert("c", "notice", text="seen by a phone, the process ends before any timer")
        cursor, log = replay(transcript)[-1]["seq"], transcript.log
        again = Transcript(tmp_path, autosave=False, clock=clock)
        again.upsert("d", "user", text="more")
        lines = replay(again, after=cursor, log=log)
        assert lines[0] == {"reset": True} and "d" in ids(lines[1:])

    def test_a_save_due_after_the_closing_one_leaves_the_file_closed(self, transcript, tmp_path, clock):
        transcript.upsert("a", "user", text="hi")
        cursor = replay(transcript)[-1]["seq"]
        transcript.flush(closed=True)
        transcript.flush()      # the timer that was already waiting for the lock
        again = Transcript(tmp_path, autosave=False, clock=clock)
        assert again.log == transcript.log and replay(again, after=cursor) == []

    def test_a_save_cancels_the_one_that_was_pending(self, tmp_path, clock, monkeypatch):
        timers = []

        class Timer:
            def __init__(self, delay, body):
                self.cancelled = False
                timers.append(self)

            def start(self):
                pass

            def cancel(self):
                self.cancelled = True

        monkeypatch.setattr(transcript_module, "threading",
                            SimpleNamespace(Timer=Timer, RLock=threading.RLock, Lock=threading.Lock))
        saving = Transcript(tmp_path, clock=clock)
        saving.upsert("a", "user", text="hi")
        saving.flush(closed=True)
        assert [timer.cancelled for timer in timers] == [True]

    def test_an_unreadable_file_starts_an_empty_log(self, tmp_path, clock):
        (tmp_path / transcript_module.FILE_NAME).write_text("{not json")
        assert Transcript(tmp_path, autosave=False, clock=clock).items() == []

    @pytest.mark.parametrize("state", [
        {"log": "l", "seq": 2, "floor": 0, "removed": [], "entries": []},
        {"log": "l", "seq": 2, "floor": 0, "removed": {}, "entries": [{"seq": 1, "item": {"id": "a", "kind": "user"}}]},
    ], ids=["the removals as a list", "an item without its time"])
    def test_a_file_of_another_shape_starts_an_empty_log_too(self, tmp_path, clock, state):
        (tmp_path / transcript_module.FILE_NAME).write_text(json.dumps(state))
        transcript = Transcript(tmp_path, autosave=False, clock=clock)
        transcript.upsert("b", "user", text="hi")
        assert [item["id"] for item in transcript.items()] == ["b"]

    def test_a_save_that_cannot_be_written_leaves_the_log_in_use(self, tmp_path, clock):
        (tmp_path / "taken").write_text("a file where the folder should be")
        transcript = Transcript(tmp_path / "taken", autosave=False, clock=clock)
        transcript.upsert("a", "user", text="hi")
        transcript.flush()
        transcript.upsert("b", "user", text="more")
        assert [item["id"] for item in transcript.items()] == ["a", "b"]

    def test_a_burst_of_changes_is_one_save_and_the_next_change_is_saved_again(self, tmp_path, clock, monkeypatch):
        timers = []

        class Timer:
            def __init__(self, delay, save):
                self.save = save
                timers.append(self)

            def start(self):
                pass

            def cancel(self):
                pass

        monkeypatch.setattr(transcript_module, "threading",
                            SimpleNamespace(Timer=Timer, RLock=threading.RLock, Lock=threading.Lock))
        transcript = Transcript(tmp_path, clock=clock)
        for text in ("Hel", "Hello", "Hello there"):
            transcript.upsert("a", "assistant", html=text)
        assert len(timers) == 1 and not (tmp_path / transcript_module.FILE_NAME).exists()
        timers[0].save()
        assert Transcript(tmp_path, autosave=False, clock=clock).item("a")["html"] == "Hello there"
        transcript.remove("a")
        assert len(timers) == 2
        timers[1].save()
        assert Transcript(tmp_path, autosave=False, clock=clock).items() == []


class TestSubscription:
    def test_drain_hands_out_each_line_once(self):
        subscription = Subscription(lambda: None)
        subscription.push("a", {"n": 1})
        assert subscription.drain() == [{"n": 1}]
        assert subscription.drain() == []
