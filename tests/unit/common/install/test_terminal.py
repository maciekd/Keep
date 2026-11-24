"""The installer's terminal: what is said, what is logged, and when there is nobody to ask."""
import io

import pytest

from common import Platform
from common.install.Terminal import InstallFailed, Terminal

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
