"""PrefillTracker — the estimate behind the "Prefilling…" bar: what the log says, how far the
clock says, and which request's bar it is; and LogSource, the file here or over ssh."""
import re
import time
from unittest.mock import MagicMock, patch

import pytest

from assistant.llm.PrefillTracker import LogFormat, LogSource, PrefillTracker, parse_last


class _ToyFormat(LogFormat):
    """A toy server log: `work <cached> <remaining>` opens a request, `step <done> <total>` is a progress line."""
    def work(self, text):
        match = re.search(r'work (\d+) (\d+)', text)
        return (int(match.group(1)), int(match.group(2))) if match else None

    def progress(self, text):
        return parse_last(r'step (\d+) (\d+)', text)


def _ssh_result(returncode=0, stdout='', stderr=''):
    result = MagicMock()
    result.returncode, result.stdout, result.stderr = returncode, stdout, stderr
    return result


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

    def test_the_clock_against_the_rate_estimates_the_rest(self, tracker, log_path):
        token = tracker.begin()
        _append(log_path, "work 0 1000\n")           # 1000 tokens at 1000 t/s: a 1 s eta
        tracker._find(token).start_ts = time.monotonic() - 0.5
        assert 40.0 < tracker.progress() < 60.0
        tracker.end(token)

    def test_the_servers_progress_lines_win_over_the_clock(self, tracker, log_path):
        token = tracker.begin()
        _append(log_path, "work 0 2000\nstep 500 2000\nstep 1000 2000\n")
        tracker._find(token).start_ts = time.monotonic() - 9999   # the clock alone would say 99
        assert tracker.progress() == pytest.approx(50.0)
        tracker.end(token)

    def test_the_rate_is_measured_from_the_progress_lines_else_the_estimate(self, tracker, log_path):
        assert PrefillTracker.current_rate() is None
        token = tracker.begin()
        _append(log_path, "work 0 2000\n")
        assert tracker.rate() == 1000.0                     # nothing measured yet: the rolling estimate
        _append(log_path, "step 500 2000\n")
        tracker._find(token).start_ts = time.monotonic() - 2.0
        tracker.work(token, force=True)                      # read the progress line now
        assert 200.0 < tracker.rate() < 300.0               # 500 tokens in ~2 s, from the line's read time
        assert PrefillTracker.current_rate() == tracker.rate()
        tracker.end(token)

    def test_capped_at_99_until_the_first_token(self, tracker, log_path):
        token = tracker.begin()
        _append(log_path, "work 0 1000\n")
        tracker._find(token).start_ts = time.monotonic() - 9999
        assert tracker.progress() == 99.0
        tracker.first_token(token)
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
    def test_the_bar_follows_the_request_the_server_is_on(self, tracker, log_path):
        # The pre-heat fired at the end of one turn is often still prefilling when the next user
        # request goes out: the user waits for it, so its bar shows until it is over; the user
        # request's own workload line is then read from its own anchor, past the pre-heat's.
        log_path.write_text("work 0 1000\n")
        with patch.object(tracker.log, 'size', return_value=0):
            pre_heat = tracker.begin(interactive=False)
        user = tracker.begin()
        tracker._find(pre_heat).start_ts = time.monotonic() - 9999
        assert tracker.progress() == 99.0, "the pre-heat's prefill is what the server is doing"
        tracker.first_token(user)
        assert tracker.progress() == 99.0, "a token the queued request cannot have had changes nothing"
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
    def test_a_finished_request_refines_the_rate_by_half(self, tracker):
        tracker.observe_rate(558, 1.0)
        assert tracker.tps == pytest.approx(0.5 * 1000.0 + 0.5 * 558.0)

    def test_a_near_full_cache_hit_teaches_nothing(self, tracker):
        tracker.observe_rate(100, 0.01)
        tracker.observe_rate(500, 0.0)
        assert tracker.tps == 1000.0


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
        request.start_ts = first_seen - 2.0
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
