"""The installer's local AI: oMLX through Homebrew, the model, the settings, the service and the
proof — over fakes of Homebrew, Hugging Face and the server. Nothing here runs `brew`, downloads
or reaches a port: the conftest refuses each."""
import json
import sys
import threading

import pytest
import requests

from common import Platform
from common.install import LocalRunner as module
from common.install.LocalRunner import (CEILING_BELOW_GB, CEILING_GB, DEFAULT_PORT, FORMULA, MODEL, MODEL_REPO,
                                        MODEL_REVISION, START_WAIT_SECONDS, TAP, TAP_COMMIT, LocalModel, LocalRunner)
from common.install.Terminal import InstallFailed, Terminal

from .fakes import ScriptedTerminal

BREW = '/opt/homebrew/bin/brew'
FILES = [('config.json', 40), ('model-00001-of-00002.safetensors', 3000), ('model-00002-of-00002.safetensors', 2000)]
USERS_SETTINGS = {'version': '1.0', 'server': {'port': 8011, 'host': '127.0.0.1'}, 'auth': {'api_key': 'omlx-key-47L'},
                  'memory': {'memory_guard_tier': 'balanced', 'soft_threshold': 0.85}}


class Runner(LocalRunner):
    """A runner on a machine made of answers. `installed` is the oMLX version Homebrew has ("":
    none), `listing` what successive `GET /v1/models` answer (the last one repeats; None: nothing
    answers), `completion` the first completion's (status, words), `service` whether the launchd
    service runs. `ran` is every command with its environment; a label in `failing` fails."""

    def __init__(self, tmp_path, answers=(), installed='0.8.3', listing=(None, [MODEL]), completion=(200, ''),
                 service=False, failing=(), download_fails=None, brew_installer=0):
        terminal = ScriptedTerminal(None if answers is None else list(answers), log_path=str(tmp_path / 'install.log'))
        super().__init__(terminal, {}, home=tmp_path / 'omlx')
        self.tmp = tmp_path
        self.installed, self.listing, self.completion, self.service = installed, list(listing), completion, service
        self.failing, self.download_fails, self.brew_installer = failing, download_fails, brew_installer
        self.ran: list[tuple[list, dict | None]] = []
        self.downloads = self.completions = self.brew_installs = 0
        self.now = 0.0
        terminal.clock = lambda: self.now
        terminal.run = self.run
        self.sleep = self.slept

    def slept(self, seconds):
        self.now += seconds

    def run(self, label, argv, env=None) -> bool:
        self.ran.append((list(argv), env))
        return not any(word in label for word in self.failing)

    def output(self, argv):
        if argv[1:] == ['list', '--versions', 'omlx']:
            return (0, f'omlx {self.installed}') if self.installed else (1, '')
        if argv[1:] == ['--repo', TAP]:
            return 0, str(self.tmp / 'taps' / 'homebrew-omlx')
        if argv[1:4] == ['services', 'info', 'omlx']:
            return 0, json.dumps([{'name': 'omlx', 'running': self.service}])
        raise AssertionError(f'a command nobody staged: {argv}')

    def files(self):
        return list(FILES)

    def download(self, target):
        self.downloads += 1
        target.mkdir(parents=True, exist_ok=True)
        (target / FILES[0][0]).write_bytes(b'x' * FILES[0][1])
        if self.download_fails:
            raise self.download_fails
        for name, size in FILES[1:]:
            (target / name).write_bytes(b'x' * size)

    def models(self):
        return self.listing.pop(0) if len(self.listing) > 1 else self.listing[0]

    def complete(self):
        self.completions += 1
        return self.completion

    def install_brew(self):
        self.brew_installs += 1
        return self.brew_installer

    def brew(self, *words) -> list[list]:
        """The brew commands that ran, as their words after `brew`; with *words*, those starting so."""
        calls = [argv[1:] for argv, _ in self.ran if argv[0] == BREW]
        return [call for call in calls if call[:len(words)] == list(words)]

    def have_model(self, folder=None):
        target = (folder or self.home / 'models') / MODEL
        target.mkdir(parents=True)
        for name, size in FILES:
            (target / name).write_bytes(b'x' * size)

    def have_settings(self, settings=USERS_SETTINGS) -> str:
        self.home.mkdir(parents=True, exist_ok=True)
        text = settings if isinstance(settings, str) else json.dumps(settings, indent=4)
        self.settings_path().write_text(text)
        return text


@pytest.fixture(autouse=True)
def a_mac_with_homebrew(monkeypatch):
    monkeypatch.setattr(module, 'find_brew', lambda: BREW)
    monkeypatch.setattr(Platform, 'memory_gb', lambda: 32)
    monkeypatch.setattr(module, 'WATCH_SECONDS', 0.001)


@pytest.mark.unit
class TestAFreshMac:
    def test_everything_is_set_up_and_the_model_is_what_the_config_gets(self, tmp_path):
        runner = Runner(tmp_path, installed='')
        assert runner.set_up() == LocalModel(MODEL)
        assert all((runner.home / 'models' / MODEL / name).stat().st_size == size for name, size in FILES)
        assert runner.completions == 1

    def test_omlx_is_tapped_moved_to_the_tested_commit_built_and_pinned_in_that_order(self, tmp_path):
        runner = Runner(tmp_path, installed='')
        runner.set_up()
        commands = [argv for argv, _ in runner.ran]
        tap, checkout, install, pin = (next(i for i, argv in enumerate(commands) if words(argv))
                                       for words in (lambda a: a[1:3] == ['tap', TAP], lambda a: a[0] == 'git',
                                                     lambda a: a[1:] == ['install', FORMULA], lambda a: a[1:2] == ['pin']))
        assert tap < checkout < install < pin
        assert commands[checkout][-1] == TAP_COMMIT and str(tmp_path / 'taps' / 'homebrew-omlx') in commands[checkout]

    def test_homebrew_is_kept_from_updating_the_tap_off_that_commit(self, tmp_path):
        """Contract with Homebrew: an auto-update before the install would move the tap to its
        newest formula, and the version built would be whatever is newest that day."""
        runner = Runner(tmp_path, installed='')
        runner.set_up()
        install_env = next(env for argv, env in runner.ran if argv[1:2] == ['install'])
        assert install_env['HOMEBREW_NO_AUTO_UPDATE'] == '1'
        assert install_env.get('PATH'), "the build needs the user's environment, not an empty one"

    def test_a_build_that_fails_stops_before_anything_is_started(self, tmp_path):
        runner = Runner(tmp_path, installed='', failing=('Build',))
        with pytest.raises(InstallFailed):
            runner.set_up()
        assert runner.brew('pin') == [] and runner.brew('services') == []
        assert runner.downloads == 0 and runner.completions == 0

    def test_a_pin_that_fails_is_a_warning_not_the_end(self, tmp_path):
        runner = Runner(tmp_path, installed='', failing=('pin',))
        assert runner.set_up() == LocalModel(MODEL)
        assert 'brew upgrade' in runner.terminal.said

    def test_a_new_install_restarts_the_service_so_it_starts_on_what_was_written(self, tmp_path):
        runner = Runner(tmp_path, installed='')
        runner.set_up()
        assert runner.brew('services') == [['services', 'restart', FORMULA]]


@pytest.mark.unit
class TestWhatIsAlreadyThere:
    def test_an_installed_omlx_is_kept_whatever_its_version(self, tmp_path):
        runner = Runner(tmp_path, installed='0.7.9')
        runner.set_up()
        assert runner.brew('tap') == runner.brew('install') == runner.brew('pin') == []
        assert not any(argv[0] == 'git' for argv, _ in runner.ran)
        assert '0.7.9' in runner.terminal.said

    def test_a_model_that_is_complete_is_not_downloaded_again(self, tmp_path):
        runner = Runner(tmp_path)
        runner.have_model()
        assert runner.ensure_model() is False
        assert runner.downloads == 0

    def test_a_model_cut_short_is_downloaded_on(self, tmp_path):
        runner = Runner(tmp_path)
        runner.have_model()
        (runner.home / 'models' / MODEL / FILES[2][0]).write_bytes(b'x' * 700)
        assert runner.ensure_model() is True
        assert runner.downloads == 1

    def test_the_model_is_looked_for_where_omlxs_settings_keep_models(self, tmp_path):
        runner = Runner(tmp_path)
        elsewhere = tmp_path / 'Volumes' / 'Models'
        runner.have_settings({'model': {'model_dirs': [str(elsewhere)]}, 'memory': {'memory_guard_tier': 'custom'}})
        runner.have_model(elsewhere)
        runner.set_up()
        assert runner.downloads == 0

    def test_with_nothing_new_the_service_is_started_not_restarted(self, tmp_path, monkeypatch):
        """Nothing changed for a running oMLX to pick up: whatever it is serving goes on."""
        monkeypatch.setattr(Platform, 'memory_gb', lambda: 64)
        runner = Runner(tmp_path)
        runner.have_model()
        runner.set_up()
        assert runner.brew('services') == [['services', 'start', FORMULA]]

    @pytest.mark.parametrize('settings, downloaded', [(False, True), (True, False)], ids=['a model downloaded', 'the ceiling written'])
    def test_anything_new_restarts_the_service(self, tmp_path, monkeypatch, settings, downloaded):
        monkeypatch.setattr(Platform, 'memory_gb', lambda: 32 if settings else 64)
        runner = Runner(tmp_path)
        if not downloaded:
            runner.have_model()
        runner.set_up()
        assert runner.brew('services') == [['services', 'restart', FORMULA]]

    def test_an_omlx_started_by_hand_is_not_given_a_second_one_to_fight_over_the_port(self, tmp_path):
        runner = Runner(tmp_path, listing=([MODEL],), service=False)
        runner.have_model()
        assert runner.set_up() == LocalModel(MODEL)
        assert runner.brew('services') == []
        assert 'brew services start' in runner.terminal.said, "the owner is not told how it starts at login"

    def test_the_service_already_running_is_told_to_start_as_always(self, tmp_path):
        runner = Runner(tmp_path, listing=([MODEL],), service=True)
        runner.have_model()
        runner.set_up()
        assert len(runner.brew('services')) == 1


@pytest.mark.unit
class TestTheSettings:
    """Contract with oMLX: ~/.omlx/settings.json, its `memory`, `server`, `auth` and `model`
    sections and the key names in them (0.8.3)."""

    def test_under_36_gb_a_fixed_ceiling_is_written(self, tmp_path):
        runner = Runner(tmp_path)
        assert runner.ensure_settings() is True
        assert json.loads(runner.settings_path().read_text()) == {
            'memory': {'memory_guard_tier': 'custom', 'memory_guard_custom_ceiling_gb': CEILING_GB}}

    @pytest.mark.parametrize('memory, written', [(CEILING_BELOW_GB - 1, True), (CEILING_BELOW_GB, False), (128, False)])
    def test_a_mac_with_room_keeps_omlxs_own_guard(self, tmp_path, monkeypatch, memory, written):
        monkeypatch.setattr(Platform, 'memory_gb', lambda: memory)
        runner = Runner(tmp_path)
        assert runner.ensure_settings() is written
        assert runner.settings_path().exists() is written

    def test_an_existing_file_keeps_every_key_and_gains_the_ceiling(self, tmp_path):
        runner = Runner(tmp_path)
        runner.have_settings({'server': {'port': 8011}, 'memory': {'soft_threshold': 0.85}})
        assert runner.ensure_settings() is True
        after = json.loads(runner.settings_path().read_text())
        assert after['server'] == {'port': 8011} and after['memory']['soft_threshold'] == 0.85
        assert after['memory']['memory_guard_tier'] == 'custom'

    def test_a_guard_the_user_chose_is_never_overwritten_and_they_are_told(self, tmp_path):
        runner = Runner(tmp_path)
        before = runner.have_settings()
        assert runner.ensure_settings() is False
        assert runner.settings_path().read_text() == before
        assert 'balanced' in runner.terminal.said and str(CEILING_GB) in runner.terminal.said

    def test_a_ceiling_already_set_is_left_without_a_word(self, tmp_path):
        runner = Runner(tmp_path)
        before = runner.have_settings({'memory': {'memory_guard_tier': 'custom', 'memory_guard_custom_ceiling_gb': 26}})
        assert runner.ensure_settings() is False
        assert runner.settings_path().read_text() == before and runner.terminal.said == ''

    @pytest.mark.parametrize('text', ['{"server": {"port": 80', '["not", "settings"]'], ids=['cut short', 'not an object'])
    def test_a_file_that_cannot_be_read_stops_the_setup_and_is_left_as_it_is(self, tmp_path, text):
        runner = Runner(tmp_path)
        runner.have_settings(text)
        with pytest.raises(InstallFailed):
            runner.set_up()
        assert runner.settings_path().read_text() == text
        assert runner.brew('services') == []

    def test_the_users_port_and_key_are_what_keep_is_pointed_at(self, tmp_path):
        runner = Runner(tmp_path)
        runner.have_settings()
        runner.have_model()
        assert runner.set_up() == LocalModel(MODEL, 'http://127.0.0.1:8011/v1', 'omlx-key-47L')

    def test_the_default_port_and_no_key_leave_both_out_of_the_config(self, tmp_path):
        runner = Runner(tmp_path)
        runner.have_settings({'server': {'port': DEFAULT_PORT}, 'auth': {'api_key': None}})
        runner.have_model()
        assert runner.set_up() == LocalModel(MODEL, '', '')

    @pytest.mark.parametrize('settings', [{'server': 'localhost'}, {'server': {'port': '8011'}}, {'server': {'port': -1}},
                                          {'auth': {'api_key': 47}}, {'model': {'model_dirs': [None]}}])
    def test_a_section_written_another_way_falls_back_to_omlxs_defaults(self, tmp_path, settings):
        runner = Runner(tmp_path)
        runner.have_settings(settings)
        assert (runner.port(), runner.api_key(), runner.models_dir()) == (DEFAULT_PORT, '', runner.home / 'models')


@pytest.mark.unit
class TestTheDownload:
    def test_a_download_that_fails_stops_the_setup_and_keeps_what_arrived(self, tmp_path):
        runner = Runner(tmp_path, download_fails=OSError('connection reset'))
        with pytest.raises(InstallFailed) as failure:
            runner.set_up()
        assert (runner.home / 'models' / MODEL / FILES[0][0]).exists()
        assert runner.brew('services') == [] and runner.completions == 0
        assert any('again' in note for note in failure.value.notes), "the user is not told a re-run continues it"
        assert 'connection reset' in (tmp_path / 'install.log').read_text()

    def test_the_bar_follows_the_bytes_on_disk_part_files_included(self, tmp_path):
        """The total is the hub's, the progress the folder's: a file still arriving counts."""
        target = tmp_path / 'models' / MODEL
        (target / '.cache' / 'huggingface' / 'download').mkdir(parents=True)
        (target / 'config.json').write_bytes(b'x' * 40)
        (target / '.cache' / 'huggingface' / 'download' / 'model.safetensors.incomplete').write_bytes(b'x' * 1200)
        assert LocalRunner.on_disk(target) == 1240
        assert LocalRunner.on_disk(tmp_path / 'no-such-folder') == 0

    def test_the_bar_is_told_how_far_the_download_is_while_it_runs(self, tmp_path):
        runner = Runner(tmp_path)
        seen, arrived, looked = [], threading.Event(), threading.Event()

        class Bar:
            def update(self, done):
                seen.append(done)
                if done:
                    looked.set()

            def close(self):
                seen.append('closed')

        def download(target):
            target.mkdir(parents=True)
            (target / FILES[1][0]).write_bytes(b'x' * 1000)
            arrived.set()
            assert looked.wait(5), 'nobody looked at the folder while the download ran'
        runner.download = download
        runner.terminal.bar = lambda label, total: seen.append(total) or Bar()
        runner.ensure_model()
        assert seen[0] == sum(size for _, size in FILES)
        assert 1000 in seen and seen[-1] == 'closed'

    def test_what_is_here_of_the_model_is_counted_in_whole_gb(self, tmp_path, monkeypatch):
        monkeypatch.setattr(module, 'GB', 1000)
        runner = Runner(tmp_path)
        assert runner.downloaded_gb() == 0
        runner.have_model()
        assert runner.downloaded_gb() == sum(size for _, size in FILES) // 1000


@pytest.mark.unit
class TestTheProof:
    def test_the_server_is_waited_for_until_it_lists_the_model(self, tmp_path):
        runner = Runner(tmp_path, listing=(None, None, [], ['another-model'], [MODEL]))
        runner.have_model()
        runner.set_up()
        # the first answer went to the question whether an oMLX runs already; three more were not the model's
        assert runner.now == 3 and runner.completions == 1

    def test_nothing_answering_gives_up_after_the_wait_and_names_the_port(self, tmp_path):
        runner = Runner(tmp_path, listing=(None,))
        runner.have_settings()
        runner.have_model()
        with pytest.raises(InstallFailed) as failure:
            runner.set_up()
        assert '8011' in failure.value.reason
        assert START_WAIT_SECONDS <= runner.now <= START_WAIT_SECONDS + 1 and runner.completions == 0

    def test_a_port_answered_by_something_else_is_said_with_what_it_serves(self, tmp_path):
        runner = Runner(tmp_path, listing=(['llama-3-8b', 'nomic-embed'],), service=True)
        runner.have_model()
        with pytest.raises(InstallFailed) as failure:
            runner.set_up()
        said = failure.value.reason + ' '.join(failure.value.notes)
        assert str(DEFAULT_PORT) in said and 'llama-3-8b' in said
        assert runner.completions == 0

    def test_a_model_that_does_not_load_fails_with_the_servers_own_words(self, tmp_path):
        runner = Runner(tmp_path, completion=(507, 'memory ceiling exceeded'))
        runner.have_model()
        with pytest.raises(InstallFailed) as failure:
            runner.set_up()
        assert '507' in failure.value.reason and 'memory ceiling exceeded' in failure.value.reason

    def test_a_server_that_hangs_up_on_the_completion_fails_too(self, tmp_path):
        runner = Runner(tmp_path, completion=(0, 'ConnectionError'))
        runner.have_model()
        with pytest.raises(InstallFailed):
            runner.set_up()


@pytest.mark.unit
class TestHomebrew:
    def test_without_homebrew_and_without_a_terminal_its_installer_is_not_started(self, tmp_path, monkeypatch):
        monkeypatch.setattr(module, 'find_brew', lambda: '')
        runner = Runner(tmp_path, answers=None)
        with pytest.raises(InstallFailed) as failure:
            runner.set_up()
        assert runner.brew_installs == 0 and 'Homebrew' in failure.value.reason

    def test_homebrews_installer_runs_and_the_brew_it_left_is_used(self, tmp_path, monkeypatch):
        found = iter(['', '/opt/homebrew/bin/brew'])
        monkeypatch.setattr(module, 'find_brew', lambda: next(found))
        runner = Runner(tmp_path)
        assert runner.ensure_brew() == '/opt/homebrew/bin/brew'
        assert runner.brew_installs == 1

    @pytest.mark.parametrize('status, found', [(1, '/opt/homebrew/bin/brew'), (0, '')], ids=['it failed', 'it left no brew'])
    def test_an_installer_that_did_not_install_stops_the_setup(self, tmp_path, monkeypatch, status, found):
        answers = iter(['', found])
        monkeypatch.setattr(module, 'find_brew', lambda: next(answers))
        with pytest.raises(InstallFailed):
            Runner(tmp_path, brew_installer=status).ensure_brew()

    def test_brew_is_found_by_its_own_path_when_no_path_names_it(self, tmp_path, monkeypatch):
        """A Homebrew installed a minute ago is on no PATH of this process."""
        monkeypatch.undo()
        brew = tmp_path / 'brew'
        brew.write_text('#!/bin/sh\n')
        brew.chmod(0o755)
        monkeypatch.setattr(module.shutil, 'which', lambda name: None)
        monkeypatch.setattr(module, 'BREW_PATHS', (str(tmp_path / 'no-such-brew'), str(brew)))
        assert module.find_brew() == str(brew)
        monkeypatch.setattr(module, 'BREW_PATHS', (str(tmp_path / 'no-such-brew'),))
        assert module.find_brew() == ''

    def test_the_installer_is_fetched_and_run_on_the_terminal(self, tmp_path, monkeypatch):
        """Homebrew's installer asks for the password: it gets the terminal, not the installer's
        closed stdin."""
        tty = tmp_path / 'tty'
        tty.write_text('')
        ran = {}

        class Fetched:
            text = 'echo "installing Homebrew"\n'

            def raise_for_status(self):
                pass

        def run(argv, stdin=None, **kwargs):
            ran.update(argv=argv, script=open(argv[-1]).read(), stdin=stdin.name)
            return type('Done', (), {'returncode': 3})()
        monkeypatch.setattr(Terminal, 'TTY', str(tty))
        monkeypatch.setattr(requests, 'get', lambda url, **kwargs: Fetched())
        monkeypatch.setattr(module.subprocess, 'run', run)
        assert LocalRunner(ScriptedTerminal(['']), {}, tmp_path).install_brew() == 3
        assert ran['script'] == Fetched.text and ran['stdin'] == str(tty) and ran['argv'][0].endswith('bash')

    def test_an_installer_that_cannot_be_fetched_stops_the_setup(self, tmp_path, monkeypatch):
        def unreachable(url, **kwargs):
            raise requests.ConnectionError('no route')
        monkeypatch.setattr(requests, 'get', unreachable)
        with pytest.raises(InstallFailed):
            LocalRunner(ScriptedTerminal(['']), {}, tmp_path).install_brew()


class Answer:
    def __init__(self, status=200, body=None, text=''):
        self.status_code, self.body, self.text = status, body, text

    def json(self):
        if self.body is None:
            raise ValueError('no JSON')
        return self.body


@pytest.mark.unit
class TestTheWire:
    """The runner's own calls, over a `requests` of the test's: the address and the key are the
    ones oMLX's settings hold."""

    def runner(self, tmp_path, settings=None) -> LocalRunner:
        runner = LocalRunner(ScriptedTerminal(), {}, tmp_path)
        if settings:
            (tmp_path / 'settings.json').write_text(json.dumps(settings))
        return runner

    def test_the_models_are_asked_of_the_port_in_the_settings_with_the_key_in_them(self, tmp_path, monkeypatch):
        asked = {}

        def get(url, headers=None, **kwargs):
            asked.update(url=url, headers=headers)
            return Answer(body={'data': [{'id': MODEL}, {'id': 'nomic-embed'}]})
        monkeypatch.setattr(requests, 'get', get)
        assert self.runner(tmp_path, USERS_SETTINGS).models() == [MODEL, 'nomic-embed']
        assert asked['url'].startswith('http://127.0.0.1:8011/') and 'omlx-key-47L' in asked['headers']['Authorization']

    def test_a_fresh_install_sends_no_key(self, tmp_path, monkeypatch):
        asked = {}
        monkeypatch.setattr(requests, 'get', lambda url, headers=None, **kwargs: asked.update(headers=headers) or Answer(body={'data': []}))
        assert self.runner(tmp_path).models() == []
        assert 'Authorization' not in asked['headers']

    def test_nothing_on_the_port_is_told_apart_from_something_that_is_not_omlx(self, tmp_path, monkeypatch):
        def refused(url, **kwargs):
            raise requests.ConnectionError('refused')
        monkeypatch.setattr(requests, 'get', refused)
        assert self.runner(tmp_path).models() is None
        for answer in (Answer(text='<html>'), Answer(body=['a', 'list']), Answer(body={'data': None})):
            monkeypatch.setattr(requests, 'get', lambda url, **kwargs: answer)
            assert self.runner(tmp_path).models() == []

    def test_the_proof_is_one_token_of_the_installed_model(self, tmp_path, monkeypatch):
        sent = {}

        def post(url, json=None, **kwargs):
            sent.update(url=url, **json)
            return Answer()
        monkeypatch.setattr(requests, 'post', post)
        assert self.runner(tmp_path, USERS_SETTINGS).complete() == (200, '')
        assert sent['url'] == 'http://127.0.0.1:8011/v1/chat/completions'
        assert sent['model'] == MODEL and sent['max_tokens'] == 1

    def test_a_refusal_comes_back_with_the_servers_words_and_a_dead_line_as_no_status(self, tmp_path, monkeypatch):
        monkeypatch.setattr(requests, 'post', lambda url, **kwargs: Answer(507, text=' memory ceiling exceeded \n'))
        assert self.runner(tmp_path).complete() == (507, 'memory ceiling exceeded')

        def dropped(url, **kwargs):
            raise requests.ReadTimeout('too slow')
        monkeypatch.setattr(requests, 'post', dropped)
        assert self.runner(tmp_path).complete()[0] == 0


@pytest.mark.unit
class TestTheHub:
    """Over a `huggingface_hub` of the test's. The repository has been re-quantized in place: what
    is listed and what is downloaded must be the one commit."""

    def test_the_files_are_those_of_the_pinned_commit(self, tmp_path, monkeypatch):
        import huggingface_hub
        asked = {}

        class Hub:
            def model_info(self, repo, revision=None, files_metadata=False):
                asked.update(repo=repo, revision=revision, sizes=files_metadata)
                sibling = lambda name, size: type('Sibling', (), {'rfilename': name, 'size': size})()
                return type('Info', (), {'siblings': [sibling('config.json', 40), sibling('.gitattributes', None)]})()
        monkeypatch.setattr(huggingface_hub, 'HfApi', Hub)
        assert LocalRunner(ScriptedTerminal(), {}, tmp_path).files() == [('config.json', 40), ('.gitattributes', 0)]
        assert asked == {'repo': MODEL_REPO, 'revision': MODEL_REVISION, 'sizes': True}

    def test_a_hub_that_does_not_answer_stops_the_setup(self, tmp_path, monkeypatch):
        import huggingface_hub

        class Hub:
            def model_info(self, *args, **kwargs):
                raise OSError('no route to host')
        monkeypatch.setattr(huggingface_hub, 'HfApi', Hub)
        with pytest.raises(InstallFailed):
            LocalRunner(ScriptedTerminal(), {}, tmp_path).files()

    def test_the_download_is_of_that_commit_into_the_folder_omlx_serves(self, tmp_path, monkeypatch):
        import huggingface_hub
        asked = {}
        monkeypatch.setattr(huggingface_hub, 'snapshot_download',
                            lambda repo, revision=None, local_dir=None, **kwargs: asked.update(repo=repo, revision=revision, into=local_dir))
        LocalRunner(ScriptedTerminal(), {}, tmp_path).download(tmp_path / 'models' / MODEL)
        assert asked == {'repo': MODEL_REPO, 'revision': MODEL_REVISION, 'into': str(tmp_path / 'models' / MODEL)}


@pytest.mark.unit
class TestTheCommandsThatOnlyAnswer:
    def test_output_is_the_status_and_what_was_printed(self, tmp_path):
        runner = LocalRunner(ScriptedTerminal(), {}, tmp_path)
        assert runner.output([sys.executable, '-c', 'print(" omlx 0.8.3 ")']) == (0, 'omlx 0.8.3')
        assert runner.output([sys.executable, '-c', 'import sys; sys.exit(4)']) == (4, '')

    def test_a_command_that_is_not_there_is_a_failure_not_a_crash(self, tmp_path):
        assert LocalRunner(ScriptedTerminal(), {}, tmp_path).output([str(tmp_path / 'no-such-brew'), 'list'])[0] != 0

    @pytest.mark.parametrize('said, runs', [('[{"name": "omlx", "running": true}]', True), ('[{"name": "omlx", "running": false}]', False),
                                            ('[]', False), ('Error: no such formula', False), ('{"running": true}', False)])
    def test_whether_the_service_runs_is_read_from_homebrews_answer(self, tmp_path, said, runs):
        """Contract with `brew services info --json`."""
        runner = LocalRunner(ScriptedTerminal(), {}, tmp_path)
        runner.output = lambda argv: (0, said)
        assert runner.service_runs(BREW) is runs


@pytest.mark.unit
def test_the_model_keeps_qwen_in_the_name_keep_reads_the_call_syntax_from():
    """Contract with assistant/commands/formats: a model called qwen… writes Qwen's call syntax."""
    assert 'qwen' in MODEL.lower()
