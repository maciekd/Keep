"""PrefillTracker — the estimate behind the "Reading…" bar: what the log says, how far the
clock says, and which request's bar it is; and LogSource, the file here or over ssh."""
import re
from unittest.mock import MagicMock, patch

import pytest

from assistant.llm.PrefillTracker import LogFormat, LogSource, PrefillTracker, common_prefix_length, parse_last


class _ToyFormat(LogFormat):
    """A toy server log: `work <cached> <remaining>` opens a request, `step <done> <total>` is a progress
    line, `finished [<seconds>s]` says the request the server is on is past its prefill."""
    def work(self, text):
        match = re.search(r'work (\d+) (\d+)', text)
        return (int(match.group(1)), int(match.group(2))) if match else None

    def progress(self, text):
        return parse_last(r'step (\d+) (\d+)', text)

    def done(self, text):
        at = text.find('finished')
        return None if at < 0 else text.find('\n', at) + 1

    def duration(self, text):
        match = re.search(r'finished (\d+\.\d+)s', text)
        return float(match.group(1)) if match else None

    def skip(self, text, requests):
        for _ in range(requests):
            at = self.done(text)
            if at is None:
                return ''
            text = text[at:]
        return text


def _ssh_result(returncode=0, stdout='', stderr=''):
    result = MagicMock()
    result.returncode, result.stdout, result.stderr = returncode, stdout, stderr
    return result


@pytest.fixture(autouse=True)
def every_prefill_drawn():
    """The tests below are about the bar's numbers, on workloads short against the seed rate."""
    with patch.object(PrefillTracker, "MIN_BAR_SECONDS", 0.0):
        yield


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "toy.log"


@pytest.fixture
def tracker(log_path):
    """A throwaway tracker on a tmp log, out of the registry again afterwards."""
    t = PrefillTracker("Toy", _ToyFormat(), seed_tps=1000.0)
    t.use_log(str(log_path))
    yield t
    PrefillTracker._trackers.remove(t)


def _append(path, text):
    with open(path, 'a') as f:
        f.write(text)


class TestEstimate:
    def test_none_while_idle_and_again_after_the_request(self, tracker):
        assert tracker.progress() is None
        tracker.end(tracker.begin())
        assert tracker.progress() is None

    def test_zero_before_the_server_logs_the_workload(self, tracker, log_path):
        log_path.write_text("nothing useful yet\n")
        token = tracker.begin()
        assert tracker.progress() == 0.0
        tracker.end(token)

    def test_a_full_cache_hit_is_complete_at_once(self, tracker, log_path):
        token = tracker.begin()
        _append(log_path, "work 100 0\n")
        assert tracker.progress() == 100.0
        tracker.end(token)

    @pytest.mark.parametrize("remaining, drawn", [(500, False), (5000, True)])   # at 1000 t/s: 0.5 s, 5 s
    def test_only_a_prefill_long_enough_to_see_is_drawn(self, tracker, log_path, remaining, drawn, prefill_clock):
        token = tracker.begin()
        _append(log_path, f"work 0 {remaining}\n")
        tracker.work(token, force=True)
        tracker._find(token).work_ts = prefill_clock.monotonic() - 0.3
        with patch.object(PrefillTracker, "MIN_BAR_SECONDS", 2.0):
            assert (tracker.progress() > 0.0) == drawn
        tracker.end(token)

    def test_the_clock_against_the_rate_estimates_the_rest(self, tracker, log_path, prefill_clock):
        token = tracker.begin()
        _append(log_path, "work 0 1000\n")           # 1000 tokens at 1000 t/s: a 1 s eta
        tracker.work(token, force=True)
        tracker._find(token).work_ts = prefill_clock.monotonic() - 0.5
        assert tracker.progress() == pytest.approx(50.0)
        tracker.end(token)

    def test_the_clock_starts_at_the_workload_line_not_at_the_send(self, tracker, log_path, prefill_clock):
        # A request sent while the server is busy waits before its prefill starts; counted as
        # prefill, that wait ran the bar up to 99 and the engine's first progress line then
        # pulled it back.
        token = tracker.begin()
        tracker._find(token).start_ts = prefill_clock.monotonic() - 9999   # a long wait behind another request
        _append(log_path, "work 0 1000\n")
        assert tracker.progress() < 10.0
        tracker.end(token)

    def test_the_servers_progress_lines_win_over_the_clock(self, tracker, log_path, prefill_clock):
        token = tracker.begin()
        _append(log_path, "work 0 2000\nstep 500 2000\nstep 1000 2000\n")
        tracker.work(token, force=True)
        tracker._find(token).work_ts = prefill_clock.monotonic() - 9999   # the clock alone would say 99
        assert tracker.progress() == pytest.approx(50.0)
        tracker.end(token)

    def test_the_rate_is_measured_from_the_progress_lines_else_the_estimate(self, tracker, log_path, prefill_clock):
        assert PrefillTracker.current_rate() is None
        token = tracker.begin()
        _append(log_path, "work 0 2000\n")
        assert tracker.rate() == 1000.0                     # nothing measured yet: the rolling estimate
        _append(log_path, "step 500 2000\n")
        tracker._find(token).work_ts = prefill_clock.monotonic() - 2.0
        tracker.work(token, force=True)                      # read the progress line now
        assert tracker.rate() == pytest.approx(250.0)       # 500 tokens in 2 s, from the line's read time
        assert PrefillTracker.current_rate() == tracker.rate()
        tracker.end(token)

    def test_an_estimate_that_ran_out_draws_no_bar_until_the_first_token(self, tracker, log_path, prefill_clock):
        # The clock said the prefill is over and the server said nothing: a bar parked at its
        # end would claim what nobody knows.
        token = tracker.begin()
        _append(log_path, "work 0 1000\n")
        tracker.work(token, force=True)
        tracker._find(token).work_ts = prefill_clock.monotonic() - 9999
        assert tracker.progress() == 0.0
        tracker.first_token(token)
        assert tracker.progress() == 100.0
        tracker.end(token)

    def test_the_log_saying_the_prompt_is_done_ends_the_prefill_without_a_token(self, tracker, log_path, prefill_clock):
        # A non-streamed request (the memory curation) sends nothing until its whole reply is
        # written, so the log is the only thing that can say its prefill is over.
        token = tracker.begin()
        _append(log_path, "work 0 1000\n")
        tracker.work(token, force=True)
        tracker._find(token).work_ts = prefill_clock.monotonic() - 9999
        assert tracker.progress() == 0.0
        _append(log_path, "finished\n")
        tracker._find(token).read_ts = 0.0
        assert tracker.progress() == 100.0
        tracker.end(token)

    def test_an_unreachable_log_disables_the_bar_for_the_request(self, tracker, log_path):
        # A None anchor (ssh down at request start) must not scan the log from byte 0 later.
        with patch.object(tracker.log, 'size', return_value=None):
            token = tracker.begin()
        assert tracker._find(token).anchor is None
        _append(log_path, "work 0 1000\n")
        assert tracker.work(token, force=True) is None
        assert tracker.progress() == 0.0
        tracker.end(token)


class TestQueue:
    def test_the_bar_follows_the_request_the_server_is_on(self, tracker, log_path, prefill_clock):
        # The pre-heat fired at the end of one turn is often still prefilling when the next user
        # request goes out: the user waits for it, so its bar shows until it is over; the user
        # request's own workload line is then read from its own anchor, past the pre-heat's.
        log_path.write_text("work 0 1000\n")
        with patch.object(tracker.log, 'size', return_value=0):
            pre_heat = tracker.begin(interactive=False)
        user = tracker.begin()
        tracker.work(pre_heat, force=True)
        prefill_clock.now += 0.5   # half of the 1000 tokens at the tracker's 1000 t/s
        assert tracker.progress() == pytest.approx(50.0), "the pre-heat's prefill is what the server is doing"
        tracker.first_token(user)
        assert tracker.progress() == pytest.approx(50.0), "a token the queued request cannot have had changes nothing"
        tracker._find(user).first_token = False
        tracker.end(pre_heat)
        assert tracker.progress() == 0.0, "the user request's workload is not logged yet"
        _append(log_path, "work 0 0\n")
        tracker._find(user).read_ts = 0.0   # past the poll throttle
        assert tracker.progress() == 100.0
        tracker.first_token(user)
        late = tracker.begin(interactive=False)
        assert tracker.progress() == 0.0, "queued behind the user's generation: nothing to draw yet"
        tracker.end(pre_heat)   # a second end of a finished request is a no-op
        tracker.end(late)
        assert tracker.progress() == 100.0
        tracker.end(user)
        assert tracker.progress() is None

    def test_a_reply_queued_behind_a_generating_request_reads_as_queued(self, tracker, log_path):
        # The memory curation holds the server for minutes after its prefill; a user turn that
        # arrives meanwhile waits for that generation, not for any prefill.
        log_path.write_text("work 0 1000\n")
        with patch.object(tracker.log, 'size', return_value=0):
            curation = tracker.begin()
        assert not tracker.queued(), "the curation itself waits for nothing"
        user = tracker.begin()
        assert not tracker.queued(), "the request ahead is still prefilling: that is the bar's business"
        _append(log_path, "finished\n")
        tracker._find(curation).read_ts = 0.0
        assert tracker.queued() and PrefillTracker.currently_queued()
        assert tracker.progress() == 0.0, "the curation's done line is not the user's: nothing of theirs is logged yet"
        tracker.end(curation)
        assert not tracker.queued()
        tracker.end(user)
        assert not PrefillTracker.currently_queued()

    def test_a_background_request_past_its_first_token_queues_the_user_too(self, tracker):
        photo = tracker.begin(interactive=False)
        tracker.first_token(photo)
        user = tracker.begin()
        assert tracker.queued()
        tracker.end(photo)
        tracker.end(user)

    def test_current_progress_is_the_interactive_requests_across_backends(self, tracker, tmp_path):
        other = PrefillTracker("Other", _ToyFormat(), seed_tps=1000.0)
        other.use_log(str(tmp_path / "other.log"))
        try:
            assert PrefillTracker.current_progress() is None
            background = other.begin(interactive=False)
            user = tracker.begin()
            with patch.object(other, 'progress', return_value=10.0), patch.object(tracker, 'progress', return_value=60.0):
                assert PrefillTracker.current_progress() == 60.0
                tracker.end(user)
                assert PrefillTracker.current_progress() == 10.0, "a server held by a background request is still what the next reply waits for"
                other.end(background)
                assert PrefillTracker.current_progress() is None
        finally:
            PrefillTracker._trackers.remove(other)


class TestRate:
    def test_the_first_measured_rate_replaces_the_seed_and_the_next_refine_it_by_half(self, tracker):
        # The seed is a guess at a machine the code has never seen: averaged in, a seed ten
        # times too fast took several replies to wash out.
        tracker.observe_rate(3000, 5.0)
        assert tracker.tps == pytest.approx(600.0)
        tracker.observe_rate(4000, 5.0)
        assert tracker.tps == pytest.approx(0.5 * 600.0 + 0.5 * 800.0)

    def test_the_prefill_seconds_are_the_done_lines_else_the_clock_from_the_workload_line(self, tracker, log_path, prefill_clock):
        # Never the time since the send: a request that waited behind another's reply taught
        # the estimate 1.5 t/s that way.
        token = tracker.begin()
        tracker._find(token).start_ts = prefill_clock.monotonic() - 9999
        _append(log_path, "work 0 1000\nfinished 2.5s\n")
        assert tracker.prefill_seconds(token) == 2.5
        tracker.end(token)

        token = tracker.begin()
        tracker._find(token).start_ts = prefill_clock.monotonic() - 9999
        _append(log_path, "work 0 1000\n")
        tracker.work(token, force=True)
        tracker._find(token).work_ts = prefill_clock.monotonic() - 3.0
        tracker.first_token(token)
        assert tracker.prefill_seconds(token) == pytest.approx(3.0)
        tracker.end(token)

    def test_a_server_that_says_nothing_is_timed_from_the_send_unless_the_request_waited(self, tracker, prefill_clock):
        token = tracker.begin()
        tracker._find(token).head_ts = prefill_clock.monotonic() - 4.0
        tracker.first_token(token)
        assert tracker.prefill_seconds(token) == pytest.approx(4.0)

        behind = tracker.begin()
        tracker.end(token)
        tracker._find(behind).head_ts = prefill_clock.monotonic() - 4.0
        tracker.first_token(behind)
        assert tracker.prefill_seconds(behind) is None, "its clock holds the wait behind the request ahead"
        tracker.end(behind)

        pre_heat = tracker.begin(interactive=False)
        tracker._find(pre_heat).head_ts = prefill_clock.monotonic() - 2.0
        assert tracker.prefill_seconds(pre_heat) == pytest.approx(2.0), "no first token: until now"
        tracker.end(pre_heat)

    def test_a_small_prefill_teaches_nothing_because_the_requests_fixed_start_is_most_of_it(self, tracker):
        tracker.observe_rate(135, 0.6)
        tracker.observe_rate(362, 1.05)
        tracker.observe_rate(5000, 0.0)
        assert tracker.tps == 1000.0


class TestWindow:
    """A request's view of the log opens at the log's end when it was sent, so the lines the
    request ahead logs after that land in it and must be skipped."""

    def test_the_request_ahead_finishing_in_the_window_is_skipped_after_its_first_token(self, tracker, log_path,
                                                                                          prefill_clock):
        ahead = tracker.begin()
        _append(log_path, "work 0 1000\n")
        mine = tracker.begin()                                 # sent while the one ahead prefills
        _append(log_path, "step 1000 1000\nfinished 1.0s\n")   # the one ahead's last lines
        tracker.first_token(ahead)
        assert tracker.progress() == 0.0, "the one ahead's 100% line is not mine; my workload is not logged yet"
        _append(log_path, "work 0 4000\nstep 1000 4000\n")
        tracker._find(mine).read_ts = 0.0
        assert tracker.progress() == pytest.approx(25.0)
        _append(log_path, "finished 4.0s\n")
        tracker._find(mine).read_ts = 0.0
        assert tracker.progress() == 100.0
        assert tracker.prefill_seconds(mine) == 4.0
        tracker.end(ahead)
        tracker.end(mine)

    def test_a_done_line_the_log_showed_is_skipped_only_by_the_requests_sent_before_it(self, tracker, log_path,
                                                                                        prefill_clock):
        # A non-streamed request ahead never sends a token: its done line is seen by a poll,
        # and the log's offset says which later requests have it in their window.
        ahead = tracker.begin()
        _append(log_path, "work 0 1000\n")
        before = tracker.begin()
        _append(log_path, "finished 1.0s\n")
        after = tracker.begin()
        assert tracker.queued() is True                         # the poll of the one ahead sees its done line
        tracker.end(ahead)
        _append(log_path, "work 0 4000\nstep 1000 4000\n")
        assert tracker.progress() == pytest.approx(25.0), "the request sent before the done line skips it"
        _append(log_path, "finished 4.0s\n")
        tracker.end(before)                                     # non-streamed, never polled for its done line: the end reads it
        _append(log_path, "work 0 2000\nstep 1000 2000\n")
        assert tracker.progress() == pytest.approx(50.0), "the request sent after it skips only the one in between"
        tracker.end(after)


class TestPolling:
    def test_polls_inside_the_interval_serve_the_last_read_and_force_reads_now(self, tracker, log_path):
        # Over ssh every read is a round trip and the TUI polls per frame.
        log_path.write_text("work 0 1000\n")
        token = tracker.begin()
        with patch.object(tracker.log, 'read_from', wraps=tracker.log.read_from) as read:
            tracker.progress()
            tracker.progress()
            tracker.progress()
            assert read.call_count == 1
            tracker.work(token, force=True)
            assert read.call_count == 2
        tracker.end(token)

    def test_the_workload_is_read_once_then_kept(self, tracker, log_path):
        token = tracker.begin()
        _append(log_path, "work 0 1000\n")
        assert tracker.work(token, force=True) == (0, 1000)
        log_path.write_text("work 5 5\n")   # the file changes under us: the latched line stands
        assert tracker.work(token, force=True) == (0, 1000)
        tracker.end(token)

    def test_the_workload_line_is_told_once_to_the_callback(self, tracker, log_path):
        """The backend's cache-break report runs on the first poll that reads the line, so the
        Home tab's row lands while the bar is drawn; later polls keep the latched line to themselves."""
        told = []
        token = tracker.begin(on_work=lambda cached, remaining: told.append((cached, remaining)))
        assert tracker.work(token, force=True) is None and told == []
        _append(log_path, "work 300 700\n")
        assert tracker.work(token, force=True) == (300, 700)
        tracker.work(token, force=True)
        tracker.progress()
        assert told == [(300, 700)]
        tracker.end(token)

    def test_a_failing_workload_callback_costs_nothing_but_itself(self, tracker, log_path):
        def boom(cached, remaining):
            raise RuntimeError("boom")
        token = tracker.begin(on_work=boom)
        _append(log_path, "work 0 1000\n")
        assert tracker.work(token, force=True) == (0, 1000)
        assert tracker.progress() is not None
        tracker.end(token)

    def test_log_reads_run_outside_the_lock(self, tracker, log_path):
        # first_token takes the lock per streamed token and the TUI per frame; a read over ssh
        # can take seconds, so it must never run under it.
        log_path.write_text("work 0 1000\nstep 1 1000\n")
        token = tracker.begin(interactive=True)
        tracker._find(token).anchor = 0

        def read_with_lock_free(offset):
            acquired = tracker._lock.acquire(blocking=False)
            assert acquired, "log read must not run under the tracker's lock"
            tracker._lock.release()
            return log_path.read_text()

        with patch.object(tracker.log, 'read_from', side_effect=read_with_lock_free):
            assert tracker.work(token, force=True) == (0, 1000)
            assert tracker._find(token).progress == (1, 1000)
        tracker.end(token)


class TestLocalLogSource:
    def test_reads_the_file_from_an_offset_and_expands_the_home_tilde(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "server.log").write_text("old line\nnew line\n")
        source = LogSource("Toy", "~/server.log")
        assert source.size() == len("old line\nnew line\n")
        assert source.read_from(len("old line\n")) == "new line\n"
        assert source.read_from(source.size()) is None, "nothing past EOF"

    def test_a_missing_file_is_size_zero_not_unreachable(self, tmp_path):
        source = LogSource("Toy", str(tmp_path / "absent.log"))
        assert source.size() == 0
        assert source.read_from(0) is None


@patch('assistant.llm.PrefillTracker.subprocess.run')
class TestSshLogSource:
    def test_size_parses_wc_output(self, mock_run):
        mock_run.return_value = _ssh_result(stdout='   1234\n')
        assert LogSource("Toy", "/srv/server.log", ssh_host="studio.local").size() == 1234

    def test_size_zero_when_the_remote_file_is_missing(self, mock_run):
        mock_run.return_value = _ssh_result(stdout='MISSING\n')
        assert LogSource("Toy", "/srv/server.log", ssh_host="studio.local").size() == 0

    def test_read_from_tails_at_the_offset(self, mock_run):
        mock_run.return_value = _ssh_result(stdout="work 0 1000\n")
        assert LogSource("Toy", "/srv/server.log", ssh_host="studio.local").read_from(100) == "work 0 1000\n"
        remote_cmd = mock_run.call_args[0][0][-1]
        assert 'tail -c +101' in remote_cmd, "tail offsets are 1-based"
        assert "/srv/server.log" in remote_cmd

    def test_a_home_relative_path_is_left_to_the_remote_shells_home(self, mock_run):
        # A quoted '~/x' would be a literal directory named ~ over there.
        mock_run.return_value = _ssh_result(stdout='7\n')
        LogSource("Toy", "~/Documents/ds4/ds4-server.log", ssh_host="studio.local").size()
        remote_cmd = mock_run.call_args[0][0][-1]
        assert '"$HOME"/' in remote_cmd and "'~" not in remote_cmd
        assert "Documents/ds4/ds4-server.log" in remote_cmd

    def test_a_failure_engages_a_cooldown_that_expires(self, mock_run):
        # A dead ssh connection must not stall the request path with a connect timeout per poll.
        mock_run.side_effect = [_ssh_result(returncode=255, stderr='Connection refused'), _ssh_result(stdout='42\n')]
        source = LogSource("Toy", "/srv/server.log", ssh_host="studio.local")
        assert source.size() is None
        assert source.read_from(0) is None
        assert mock_run.call_count == 1
        source._disabled_until = 0.0   # the cooldown expires
        assert source.size() == 42


class TestMeasuredRate:
    def test_a_progress_line_read_again_keeps_the_time_it_was_first_seen(self, tracker, log_path):
        """The measured rate is the tokens of the latest progress line over the time from the
        request's start to the moment that line was read: a later read of the same line must
        not move that moment, or the rate would sink at every poll while the engine is quiet,
        and a new line must."""
        token = tracker.begin()
        _append(log_path, "work 0 2000\nstep 500 2000\n")
        tracker.work(token, force=True)
        request = tracker._find(token)
        first_seen = request.progress_ts
        request.work_ts = first_seen - 2.0
        assert tracker.rate() == pytest.approx(250.0)

        request.read_ts = 0.0   # the next poll slot has come; the log has nothing new
        tracker.work(token, force=True)
        assert request.progress_ts == first_seen
        assert tracker.rate() == pytest.approx(250.0)

        _append(log_path, "step 1500 2000\n")
        tracker.work(token, force=True)
        assert request.progress_ts > first_seen
        assert request.progress == (1500, 2000)
        tracker.end(token)


class TestWireProgress:
    """A server that streams its prefill progress (llama.cpp's `prompt_progress`, oMLX's too)
    feeds the tracker through `report_progress` and needs no log."""

    @pytest.fixture
    def wire(self):
        t = PrefillTracker("Wire", None, seed_tps=1000.0)
        yield t
        t._requests.clear()
        PrefillTracker._trackers.remove(t)

    def test_the_first_chunk_is_the_workload_and_the_bar_counts_the_uncached_part(self, wire, prefill_clock):
        token = wire.begin()
        assert wire.progress() == 0.0, "nothing to draw before the server says anything"
        wire.report_progress(token, total=5000, cache=4000, processed=4500, time_ms=250.0)
        assert wire.work(token) == (4000, 1000)
        assert wire.progress() == pytest.approx(50.0), "500 of the 1000 uncached tokens, not 4500 of 5000"
        wire.report_progress(token, total=5000, cache=4000, processed=5000, time_ms=500.0)
        assert wire.progress() == 99.0, "capped until the first token confirms the prefill is over"
        wire.first_token(token)
        assert wire.progress() == 100.0
        wire.end(token)

    def test_a_full_cache_hit_is_complete_at_once(self, wire):
        token = wire.begin()
        wire.report_progress(token, total=5000, cache=5000, processed=5000, time_ms=1.0)
        assert wire.progress() == 100.0
        wire.end(token)

    def test_the_rate_and_the_duration_come_from_the_servers_clock(self, wire):
        token = wire.begin()
        wire.report_progress(token, total=5000, cache=4000, processed=4500, time_ms=250.0)
        assert wire.rate() == pytest.approx(2000.0)             # 500 uncached tokens in 0.25 s
        wire.report_progress(token, total=5000, cache=4000, processed=5000, time_ms=400.0)
        wire.first_token(token)
        assert wire.prefill_seconds(token) == pytest.approx(0.4), "the last chunk's time, not ours"
        wire.end(token)

    def test_the_workload_callback_runs_once_on_the_first_chunk(self, wire):
        told = []
        token = wire.begin(on_work=lambda cached, remaining: told.append((cached, remaining)))
        wire.report_progress(token, total=5000, cache=4000, processed=4096)
        wire.report_progress(token, total=5000, cache=4000, processed=5000)
        assert told == [(4000, 1000)]
        wire.end(token)

    def test_a_chunk_for_a_request_no_longer_in_flight_is_ignored(self, wire):
        token = wire.begin()
        wire.end(token)
        wire.report_progress(token, total=10, cache=0, processed=5)
        assert wire.progress() is None

    def test_the_bar_follows_the_request_the_server_is_on(self, wire, prefill_clock):
        # A pre-heat still prefilling when the user's request goes out: the user waits for it.
        heat = wire.begin(interactive=False)
        user = wire.begin()
        wire.report_progress(heat, total=2000, cache=0, processed=1000, time_ms=1000.0)
        assert wire.progress() == pytest.approx(50.0)
        wire.first_token(heat)
        assert wire.progress() == 0.0, "the user's request has not reported yet"
        wire.report_progress(user, total=2100, cache=2000, processed=2050, time_ms=50.0)
        # One bar for the whole wait: the pre-heat's 2000 tokens stay counted, 50 of the user's 100 are in.
        assert wire.progress() == pytest.approx(2050 / 2100 * 100)
        wire.end(heat)
        wire.end(user)


class TestForecast:
    """A server that says nothing until a whole prefill step is done (oMLX) leaves the bar on the
    tracker's own forecast: the part of the prompt a prompt the server prefilled lately shares
    is taken as cached, at that prompt's tokens per character."""

    @pytest.fixture
    def wire(self):
        t = PrefillTracker("Forecast", None, seed_tps=100.0)
        yield t
        t._requests.clear()
        PrefillTracker._trackers.remove(t)

    @staticmethod
    def _held(tracker, prompt, tokens):
        token = tracker.begin(interactive=False, prompt=prompt)
        tracker.learn(token, tokens)
        tracker.first_token(token)
        tracker.end(token)

    def test_the_shared_prefix_is_cached_at_the_learned_tokens_per_char(self, wire):
        self._held(wire, "s" * 1000, 250)                      # a quarter of a token per character
        token = wire.begin(prompt="s" * 1000 + "u" * 400)
        assert wire._find(token).forecast == (250, 100)
        wire.end(token)

    def test_the_longest_prefix_among_the_held_prompts_counts(self, wire):
        self._held(wire, "s" * 1000, 500)
        self._held(wire, "x" * 2000, 1000)                      # a newer prompt that shares nothing
        token = wire.begin(prompt="s" * 1000 + "u" * 200)
        assert wire._find(token).forecast == (500, 100)
        wire.end(token)

    def test_a_prompt_the_server_never_finished_is_not_taken_as_cached(self, wire):
        token = wire.begin(prompt="s" * 1000)
        wire.end(token)                                          # no usage: nothing learned
        token = wire.begin(prompt="s" * 1000 + "u" * 10)
        cached, remaining = wire._find(token).forecast
        assert cached == 0 and remaining > 0
        wire.end(token)

    def test_the_bar_draws_from_the_forecast_then_switches_to_the_servers_numbers(self, wire, prefill_clock):
        self._held(wire, "s" * 1000, 1000)
        token = wire.begin(prompt="s" * 1000 + "u" * 200)        # 200 tokens: 2 s at 100 t/s
        prefill_clock.now += 1.0
        assert wire.progress() == pytest.approx(50.0), "drawn before the server says anything"
        wire.report_progress(token, total=1400, cache=1000, processed=1040, time_ms=1000.0)
        assert wire.work(token) == (1000, 400)
        assert wire.progress() == pytest.approx(10.0), "the server's 40 of 400, not the forecast's clock"
        wire.end(token)

    def test_the_forecast_clock_starts_when_the_request_ahead_is_past_its_prefill(self, wire, prefill_clock):
        self._held(wire, "s" * 1000, 1000)
        ahead = wire.begin(interactive=False)
        mine = wire.begin(prompt="s" * 1000 + "u" * 200)
        prefill_clock.now += 30.0                                 # the one ahead still prefilling
        assert wire.progress() == 0.0, "the bar is the one ahead's, which has said nothing"
        wire.first_token(ahead)
        prefill_clock.now += 0.5
        assert wire.progress() == pytest.approx(25.0), "0.5 s of 2, not the 30 s since the send"
        wire.end(ahead)
        wire.end(mine)

    def test_between_two_reports_the_bar_moves_at_the_measured_rate(self, wire, prefill_clock):
        token = wire.begin()
        wire.report_progress(token, total=2000, cache=0, processed=500, time_ms=1000.0)   # 500 t/s
        assert wire.progress() == pytest.approx(25.0)
        prefill_clock.now += 1.0
        assert wire.progress() == pytest.approx(50.0)
        prefill_clock.now += 60.0
        assert wire.progress() == 99.0, "capped until the first token"
        wire.end(token)


@pytest.mark.parametrize("a, b, shared", [
    ("", "", 0), ("", "abc", 0), ("abc", "abc", 3), ("abc", "abcdef", 3),
    ("abcdef", "abc", 3), ("abX", "abY", 2), ("Xbc", "abc", 0),
])
def test_common_prefix_length(a, b, shared):
    assert common_prefix_length(a, b) == shared
