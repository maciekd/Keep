"""The installer's terminal: what is said, what is logged, and when there is nobody to ask."""
import io
import re
import sys

import pytest

from common import Palette, Platform
from common.install import Terminal as module
from common.install.Terminal import GB, InstallFailed, Terminal

from .fakes import ScriptedTerminal


class Screen(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.unit
class TestTheTerminal:
    def test_output_that_is_no_terminal_carries_no_escape_codes(self):
        terminal = ScriptedTerminal()
        terminal.step('Configuration')
        terminal.ok('done')
        terminal.warn('careful')
        terminal.failed(InstallFailed('It broke.', 'Try again.'))
        assert '\033' not in terminal.said
        assert 'It broke.' in terminal.said and 'Try again.' in terminal.said

    @pytest.mark.parametrize('env, coloured', [
        ({'TERM': 'xterm-256color'}, True),
        ({'TERM': 'xterm-256color', 'NO_COLOR': '1'}, False),
        ({'TERM': 'dumb'}, False),
        ({}, False),
    ], ids=['a terminal', 'NO_COLOR', 'a dumb terminal', 'no TERM'])
    def test_colour_is_for_a_terminal_that_did_not_decline_it(self, env, coloured):
        terminal = Terminal(out=Screen(), env=env)
        terminal.ok('done')
        assert ('\033[' in terminal.out.getvalue()) is coloured

    def test_steps_are_counted_on_from_where_the_installer_stood(self, tmp_path):
        log = tmp_path / 'install.log'
        terminal = ScriptedTerminal(log_path=str(log), step=7, steps=9)
        terminal.step('Anthropic API key')
        terminal.step('Configuration')
        assert '[8/9]' in terminal.said and '[9/9]' in terminal.said
        assert terminal.said.index('[8/9]') < terminal.said.index('[9/9]')
        assert '[8/9] Anthropic API key' in log.read_text()

    def test_a_failure_ends_the_log_with_the_step_and_the_reason(self, tmp_path):
        """Contract with install.sh's `configure`, which reads the log's last two lines for the
        install report."""
        log = tmp_path / 'install.log'
        terminal = ScriptedTerminal(log_path=str(log))
        terminal.step('Anthropic API key')
        terminal.failed(InstallFailed('No valid API key after 3 attempts.', 'Re-run the installer.'))
        assert log.read_text().splitlines()[-2:] == ['FAILED in step: Anthropic API key',
                                                     'FAILED: No valid API key after 3 attempts.']

    def test_a_note_is_shown_and_not_logged(self, tmp_path):
        """A note is where a secret may stand on the screen."""
        log = tmp_path / 'install.log'
        terminal = ScriptedTerminal(log_path=str(log))
        terminal.ok('logged')
        terminal.note('Zq4nB8cR2yH6sD0fG3jA7vK1mX5pL9wT')
        assert 'Zq4nB8cR2yH6sD0fG3jA7vK1mX5pL9wT' in terminal.said
        assert log.read_text().strip() == 'ok: logged'

    def test_without_a_log_nothing_is_written_anywhere(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ScriptedTerminal().ok('done')
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize('system, way', [(Platform.MAC, 'Terminal'), (Platform.LINUX, 'ssh -t')])
    def test_with_nobody_to_ask_the_question_names_the_way_out(self, monkeypatch, system, way):
        monkeypatch.setattr(Platform, 'current', lambda: system)
        terminal = ScriptedTerminal(answers=None)
        with pytest.raises(InstallFailed) as failure:
            terminal.ask('Anthropic API key:', secret=True)
        said = ' '.join(failure.value.notes)
        assert way in said and 'environment' in said
        assert terminal.prompts == []

    def test_a_terminal_that_cannot_be_opened_is_none_to_ask_on(self, tmp_path):
        """`ssh mini 'curl ... | bash'`: /dev/tty is there and only the open fails."""
        class NoTerminal(Terminal):
            TTY = str(tmp_path / 'no-such-tty')
        assert NoTerminal(out=io.StringIO(), env={}).interactive is False

    def test_a_terminal_that_opens_is_the_one_to_ask_on(self, tmp_path):
        class AFile(Terminal):
            TTY = str(tmp_path / 'tty')
        (tmp_path / 'tty').write_text('')
        assert AFile(out=io.StringIO(), env={}).interactive is True


def sgr(rgb) -> str:
    return '38;2;{};{};{}'.format(*rgb)


def shown(text: str) -> str:
    """*text* as it stands on the screen: without the escape codes."""
    return re.sub(r'\033\[[0-9;?]*[A-Za-z]', '', text)


def live(columns: int = 120, **env) -> Terminal:
    """A terminal that redraws in place, on a clock the test moves."""
    terminal = Terminal(out=Screen(), env={'TERM': 'xterm-256color', **env})
    terminal.now = 0.0
    terminal.clock = lambda: terminal.now
    terminal.columns = lambda: columns
    return terminal


def last_line(terminal) -> str:
    return shown(terminal.out.getvalue().split('\r')[-1])


@pytest.mark.unit
class TestTheColours:
    """The accents are the console's (common/Palette.py), as near as the terminal can draw them."""

    def green(self, **env) -> str:
        terminal = Terminal(out=Screen(), env=env)
        terminal.ok('done')
        return terminal.out.getvalue()

    @pytest.mark.parametrize('colorterm', ['truecolor', '24bit'])
    def test_a_terminal_that_says_it_has_every_colour_gets_the_palettes_own(self, colorterm):
        assert sgr(Palette.GREEN) in self.green(TERM='xterm-256color', COLORTERM=colorterm)

    def test_a_terminal_of_256_gets_the_nearest_of_them(self):
        """Terminal before macOS 26: 256 colours and no COLORTERM."""
        said = self.green(TERM='xterm-256color')
        assert f'38;5;{Palette.nearest_256(Palette.GREEN)}m' in said and '38;2;' not in said

    def test_any_other_terminal_gets_the_plain_ansi_colours(self):
        said = self.green(TERM='xterm')
        assert '\033[32m' in said and '38;' not in said

    def test_no_color_wins_over_a_terminal_that_has_every_colour(self):
        assert '\033' not in self.green(TERM='xterm-256color', COLORTERM='truecolor', NO_COLOR='1')

    def test_the_text_itself_is_left_in_the_terminals_own_colour(self):
        """A fresh Mac's Terminal is black on white: the palette's greys would vanish on it."""
        said = self.green(TERM='xterm-256color', COLORTERM='truecolor')
        assert said.index('done') > said.index('\033[0m'), "the line's text is coloured too"


@pytest.mark.unit
class TestTheBar:
    def test_it_says_how_far_how_fast_and_how_long_still(self):
        terminal = live()
        bar = terminal.bar('Qwen 3.8 27B', 20 * GB)
        bar.update(2 * GB)
        terminal.now = 10
        bar.update(3 * GB)
        line = last_line(terminal)
        assert 'Qwen 3.8 27B' in line and '15%' in line
        assert '100 MB/s' in line and '3 min' in line, line
        assert line.count('█') * 100 // (line.count('█') + line.count('░')) in range(10, 16)

    def test_the_rate_is_that_of_the_last_moments_not_of_the_whole_download(self):
        """A burst at the start (what was resumed, a cached file) must not promise minutes for hours."""
        terminal = live()
        bar = terminal.bar('Qwen 3.8 27B', 20 * GB)
        for at, done in ((0, 0), (1, 10 * GB), (20, 10.5 * GB), (30, 11 * GB)):
            terminal.now = at
            bar.update(int(done))
        assert '50 MB/s' in last_line(terminal)

    def test_a_download_that_resumes_starts_where_it_was_and_claims_no_rate_yet(self):
        terminal = live()
        terminal.bar('Qwen 3.8 27B', 20 * GB).update(9 * GB)
        line = last_line(terminal)
        assert '45%' in line and 'MB/s' not in line and 'left' not in line

    def test_a_stalled_download_promises_no_time(self):
        terminal = live()
        bar = terminal.bar('Qwen 3.8 27B', 20 * GB)
        bar.update(5 * GB)
        terminal.now = 10
        bar.update(5 * GB)
        assert 'left' not in last_line(terminal)

    def test_more_than_the_total_is_the_whole_and_no_more(self):
        terminal = live()
        terminal.bar('Qwen 3.8 27B', 20 * GB).update(23 * GB)
        line = last_line(terminal)
        assert '100%' in line and '░' not in line

    @pytest.mark.parametrize('columns', [40, 50, 64, 80, 120])
    def test_the_line_fits_the_row_it_is_redrawn_on(self, columns):
        """Redrawn with a carriage return: a line the terminal wraps leaves every earlier one behind."""
        terminal = live(columns)
        bar = terminal.bar('Qwen 3.8 27B', 20 * GB)
        bar.update(2 * GB)
        terminal.now = 10
        bar.update(3 * GB)
        line = last_line(terminal)
        assert len(line) < columns and '15%' in line and '█' in line

    def test_80_columns_hold_every_fact(self):
        terminal = live(80)
        bar = terminal.bar('Qwen 3.8 27B', int(17.5 * GB))
        bar.update(9 * GB)
        terminal.now = 10
        bar.update(int(9.4 * GB))
        line = last_line(terminal)
        assert 'of 17.5 GB' in line and 'MB/s' in line and 'left' in line, line

    def test_with_every_colour_it_fades_up_from_the_accents_deep_companion(self):
        terminal = live(COLORTERM='truecolor')
        terminal.bar('Qwen 3.8 27B', 20 * GB).update(20 * GB)
        drawn = terminal.out.getvalue()
        assert drawn.index(sgr(Palette.DEEP[Palette.CYAN])) < drawn.index(sgr(Palette.CYAN))

    def test_without_them_it_is_the_accent_alone(self):
        terminal = live()
        terminal.bar('Qwen 3.8 27B', 20 * GB).update(10 * GB)
        assert '38;2;' not in terminal.out.getvalue()

    def test_the_cursor_is_hidden_while_it_draws_and_shown_again(self):
        terminal = live()
        bar = terminal.bar('Qwen 3.8 27B', 20 * GB)
        bar.update(GB)
        assert terminal.out.getvalue().count('\033[?25l') == 1 and '\033[?25h' not in terminal.out.getvalue()
        bar.close()
        assert terminal.out.getvalue().endswith('\033[?25h')

    def test_without_a_terminal_to_redraw_on_it_is_a_line_now_and_then(self):
        terminal = ScriptedTerminal()
        bar = terminal.bar('Qwen 3.8 27B', 20 * GB)
        for hundredth in range(0, 101):
            bar.update(hundredth * 20 * GB // 100)
        bar.close()
        lines = terminal.said.splitlines()
        assert 5 <= len(lines) <= 11 and '100%' in lines[-1]
        assert '\r' not in terminal.said and '\033' not in terminal.said


@pytest.mark.unit
@pytest.mark.parametrize('seconds, said', [(20, 'minute'), (600, '10 min'), (5400, '1 h 30')])
def test_the_time_left_is_said_in_the_unit_that_fits(seconds, said):
    assert said in module.fmt_left(seconds)


@pytest.mark.unit
class TestWaiting:
    def test_it_hands_back_what_the_work_returned(self):
        assert ScriptedTerminal().wait('Loading the model', lambda: (200, 'ok')) == (200, 'ok')

    @pytest.mark.parametrize('terminal', [ScriptedTerminal, live], ids=['no terminal', 'a terminal'])
    def test_what_the_work_raised_is_the_callers_to_see(self, terminal):
        def work():
            raise InstallFailed('oMLX does not answer on port 8000.')
        with pytest.raises(InstallFailed):
            terminal().wait('Waiting for oMLX', work)

    def test_a_terminal_shows_the_label_and_gets_its_cursor_back_whatever_happened(self):
        terminal = live()

        def work():
            raise ValueError('the tap moved')
        with pytest.raises(ValueError):
            terminal.wait('Build oMLX', work)
        drawn = terminal.out.getvalue()
        assert drawn.index('\033[?25l') < drawn.rindex('\033[?25h')

    def test_without_a_terminal_the_label_is_said_once(self):
        terminal = ScriptedTerminal()
        terminal.wait('Build oMLX', lambda: None)
        assert terminal.said.count('Build oMLX') == 1 and '\r' not in terminal.said

    def test_a_label_longer_than_the_row_is_cut_to_it(self, monkeypatch):
        monkeypatch.setattr(module, 'TICK', 0.001)
        terminal = live(columns=40)
        drawn_once = []

        def work():
            while '\r' not in terminal.out.getvalue():      # until the spinner has drawn a frame
                drawn_once.append(1)
        terminal.wait('Build oMLX 0.8.3 (from source: several minutes) and everything it needs', work)
        frames = [shown(frame) for frame in terminal.out.getvalue().split('\r') if 'Build' in frame]
        assert frames and all(len(frame) < 40 for frame in frames)


@pytest.mark.unit
class TestRunningACommand:
    def terminal(self, tmp_path) -> ScriptedTerminal:
        return ScriptedTerminal(log_path=str(tmp_path / 'install.log'))

    def test_a_command_that_succeeds_is_one_line_and_its_output_is_in_the_log(self, tmp_path):
        terminal = self.terminal(tmp_path)
        assert terminal.run('Build oMLX', [sys.executable, '-c', 'print("compiling tiktoken")']) is True
        assert 'Build oMLX' in terminal.said and 'compiling tiktoken' not in terminal.said
        assert 'compiling tiktoken' in (tmp_path / 'install.log').read_text()

    def test_a_command_that_fails_shows_the_end_of_what_it_said(self, tmp_path):
        terminal = self.terminal(tmp_path)
        script = 'import sys; print("error: no rust toolchain"); sys.exit(3)'
        assert terminal.run('Build oMLX', [sys.executable, '-c', script]) is False
        assert 'no rust toolchain' in terminal.said and '3' in terminal.said

    def test_a_command_that_is_not_there_fails_like_any_other(self, tmp_path):
        terminal = self.terminal(tmp_path)
        assert terminal.run('Add oMLX to Homebrew', [str(tmp_path / 'no-such-brew'), 'tap']) is False
        assert 'Add oMLX to Homebrew' in terminal.said

    def test_the_command_gets_the_environment_it_is_given(self, tmp_path):
        terminal = self.terminal(tmp_path)
        script = 'import os; print("update=" + os.environ["HOMEBREW_NO_AUTO_UPDATE"])'
        assert terminal.run('Build oMLX', [sys.executable, '-c', script], {'HOMEBREW_NO_AUTO_UPDATE': '1'})
        assert 'update=1' in (tmp_path / 'install.log').read_text()

    def test_the_command_cannot_read_what_the_installer_was_piped(self, tmp_path):
        """curl | bash: a command that reads stdin would eat the script."""
        terminal = self.terminal(tmp_path)
        assert terminal.run('Build oMLX', [sys.executable, '-c', 'import sys; print("read=[" + sys.stdin.read() + "]")'])
        assert 'read=[]' in (tmp_path / 'install.log').read_text()

    def test_without_a_log_a_command_still_runs(self):
        assert ScriptedTerminal().run('Build oMLX', [sys.executable, '-c', 'print("x")']) is True
