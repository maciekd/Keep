"""Where a new install thinks: the gate a Mac has to pass for the local model, and the question."""
import re

import pytest

from common import Platform
from common.install import AiChoice as module
from common.install.AiChoice import CLOUD, LOCAL, AiChoice
from common.install.Terminal import InstallFailed

from .fakes import ScriptedTerminal

A_KEY = {'ANTHROPIC_API_KEY': 'sk-ant-typed-47L'}


@pytest.fixture
def machine(monkeypatch):
    """A Mac that can carry the local model, with Homebrew; a test takes one thing away."""
    def build(system=Platform.MAC, silicon=True, macos=15, memory=32, free=200, downloaded=0, brew='/opt/homebrew/bin/brew'):
        monkeypatch.setattr(Platform, 'current', lambda: system)
        monkeypatch.setattr(Platform, 'is_apple_silicon', lambda: silicon)
        monkeypatch.setattr(Platform, 'macos_version', lambda: macos)
        monkeypatch.setattr(Platform, 'memory_gb', lambda: memory)
        monkeypatch.setattr(AiChoice, 'free_gb', staticmethod(lambda: free))
        monkeypatch.setattr(AiChoice, 'runner', lambda self: type('Runner', (), {'downloaded_gb': lambda self: downloaded})())
        monkeypatch.setattr(module.LocalRunner, 'find_brew', lambda: brew)
    build()
    return build


def choice(answers=None, **env) -> AiChoice:
    return AiChoice(ScriptedTerminal(answers), env)


@pytest.mark.unit
class TestTheGate:
    def test_a_mac_that_meets_every_requirement_misses_nothing(self, machine):
        assert choice().missing() == ''

    @pytest.mark.parametrize('lacking, numbers', [
        ({'macos': 14}, ('14', '15')),
        ({'memory': 16}, ('16', '32')),
        ({'free': 21}, ('21', '30')),
    ], ids=['macOS', 'memory', 'disk'])
    def test_what_is_missing_is_named_with_this_macs_own_number_and_the_one_needed(self, machine, lacking, numbers):
        machine(**lacking)
        said = choice().missing()
        assert all(re.search(rf'\b{number}\b', said) for number in numbers), said

    def test_an_intel_mac_is_told_it_is_the_chip(self, machine):
        machine(silicon=False, memory=64)
        assert 'Intel' in choice().missing()

    def test_a_linux_server_is_told_it_is_a_mac_thing_whatever_it_has(self, machine):
        machine(system=Platform.LINUX, memory=128)
        assert 'Mac' in choice().missing()

    def test_the_first_requirement_missed_is_the_one_named(self, machine):
        machine(memory=16, free=3)
        said = choice().missing()
        assert 'memory' in said and 'disk' not in said

    def test_memory_that_cannot_be_read_is_not_enough(self, machine):
        machine(memory=0)
        assert choice().missing() != ''

    @pytest.mark.parametrize('free, downloaded, needed', [(15, 18, None), (11, 18, '12'), (11, 40, '12'), (25, 3, '27')],
                             ids=['the rest fits', 'the rest does not', 'more than the model is there', 'an interrupted download'])
    def test_what_is_already_downloaded_of_the_model_needs_no_room(self, machine, free, downloaded, needed):
        """A re-run after the download must not be refused for the room the model now takes."""
        machine(free=free, downloaded=downloaded)
        said = choice().missing()
        if needed is None:
            assert said == ''
        else:
            assert re.search(rf'\b{needed}\b', said), said


@pytest.mark.unit
class TestTheQuestion:
    @pytest.mark.parametrize('answers, chosen, asked', [
        ([''], CLOUD, 1), (['1'], CLOUD, 1), (['2'], LOCAL, 1), ([' 2 '], LOCAL, 1), (['3'], CLOUD, 1),
        (['x', '2'], LOCAL, 2), (['x', 'yes', '7'], CLOUD, 3),
    ], ids=['Enter', '1', '2', '2 with spaces', '3', 'a slip then 2', 'three slips'])
    def test_the_answer_decides(self, machine, answers, chosen, asked):
        asking = choice(answers + ['2'])      # a fourth question would be answered with the local model
        assert asking.choose() == chosen
        assert len(asking.terminal.prompts) == asked
        assert not asking.explicit

    @pytest.mark.parametrize('brew, mentioned', [('', True), ('/opt/homebrew/bin/brew', False)], ids=['no Homebrew', 'Homebrew'])
    def test_the_local_option_says_when_it_installs_homebrew(self, machine, brew, mentioned):
        machine(brew=brew)
        asking = choice(['1'])
        asking.choose()
        assert ('Homebrew' in asking.terminal.said) is mentioned

    def test_no_text_of_the_question_calls_local_private(self, machine):
        """Telemetry is on for every install: the local option promises local AI, not privacy."""
        asking = choice(['2'])
        asking.choose()
        assert 'privat' not in asking.terminal.said.lower()

    @pytest.mark.parametrize('columns', [60, 80, 200])
    def test_the_options_fit_the_terminal(self, machine, columns):
        asking = choice(['1'])
        asking.terminal.columns = lambda: columns
        asking.choose()
        assert max(len(line) for line in asking.terminal.said.splitlines()) <= columns

    @pytest.mark.parametrize('answers, env', [(['2'], A_KEY), (None, {})], ids=['a key in the environment', 'no terminal'])
    def test_an_unattended_install_is_asked_nothing_and_runs_on_claude(self, machine, answers, env):
        asking = choice(answers, **env)
        assert asking.choose() == CLOUD
        assert asking.terminal.prompts == []

    def test_a_mac_that_cannot_is_told_why_and_asked_nothing(self, machine):
        machine(memory=16)
        asking = choice(['2'])
        assert asking.choose() == CLOUD
        assert asking.terminal.prompts == []
        assert '16' in asking.terminal.said and '32' in asking.terminal.said

    def test_linux_is_told_the_way_to_local_ai_and_asked_nothing(self, machine):
        machine(system=Platform.LINUX)
        asking = choice(['2'])
        assert asking.choose() == CLOUD
        assert asking.terminal.prompts == [] and 'Keep' in asking.terminal.said


@pytest.mark.unit
class TestTheVariable:
    """KEEP_AI is a contract with install.sh's header, which documents it to the user."""

    @pytest.mark.parametrize('value', ['local', 'LOCAL', ' local '])
    def test_local_is_chosen_without_asking_even_beside_a_key(self, machine, value):
        asking = choice(['1'], KEEP_AI=value, **A_KEY)
        assert asking.choose() == LOCAL
        assert asking.terminal.prompts == [] and asking.explicit

    def test_cloud_is_chosen_without_asking(self, machine):
        asking = choice(['2'], KEEP_AI='cloud')
        assert asking.choose() == CLOUD
        assert asking.terminal.prompts == [] and not asking.explicit

    def test_any_other_value_stops_the_install_naming_it(self, machine):
        with pytest.raises(InstallFailed) as failure:
            choice(['2'], KEEP_AI='ollama').choose()
        assert 'KEEP_AI=ollama' in failure.value.reason

    @pytest.mark.parametrize('lacking, said', [({'memory': 16}, '16'), ({'system': Platform.LINUX}, 'Mac'), ({'free': 4}, '4')],
                             ids=['too little memory', 'Linux', 'no room'])
    def test_local_on_a_machine_that_cannot_stops_the_install_with_the_reason(self, machine, lacking, said):
        """Nobody is there to give a key instead: going on with Claude would only fail later."""
        machine(**lacking)
        with pytest.raises(InstallFailed) as failure:
            choice(None, KEEP_AI='local').choose()
        assert said in failure.value.reason
