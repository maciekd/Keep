"""Tests for install.sh's decision logic: the pip progress read off pip's own log, the
API key intake, the config.yaml it writes and the state machine around the checkout
(fresh clone, upgrade, refused directories).

The script's functions are sourced without the trailing `main "$@"` call and driven
one at a time. Stdout is a pipe, so the spinner (a background subshell) is off and
the output is plain text. `curl` and the terminal prompt `ask` are bash functions
sourced before the call (see test_start_sh.py for why functions, not executables);
the repository tests run real git against a throwaway origin in tmp_path.
"""
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from common.ConfigManager import REQUIRED_KEYS

INSTALL_SH = Path(__file__).resolve().parents[2] / 'install.sh'
GIT_AUTHOR = ['-c', 'user.name=Marcus', '-c', 'user.email=m@mlab.test']
BAR_RE = re.compile(r'^\[(.*)\] (.*)$')


class InstallShHarness:
    def __init__(self, tmp_path: Path):
        self.dir = tmp_path
        text = INSTALL_SH.read_text()
        assert text.endswith('main "$@"\n'), "install.sh no longer ends with the main call"
        self.fns = tmp_path / 'install_fns.sh'
        self.fns.write_text(text[:-len('main "$@"\n')])
        self.keep_dir = tmp_path / 'Keep'
        self.log = tmp_path / 'install.log'
        self.curl_calls = tmp_path / 'curl_calls'
        self.curl_code = tmp_path / 'curl_code'
        self.curl_code.write_text('200')
        self.asked = tmp_path / 'asked'
        self.answers = None

    def set_curl_code(self, code: str):
        self.curl_code.write_text(code)

    def set_answers(self, *lines: str):
        """What the user types at successive prompts; the prompt is recorded in `asked`."""
        self.answers = self.dir / 'answers'
        self.answers.write_text(''.join(f'{line}\n' for line in lines))

    def run(self, script: str, repo_url: str = '', **env) -> subprocess.CompletedProcess:
        prelude = (
            f'source "{self.fns}"\n'
            f'KEEP_DIR="{self.keep_dir}"; LOG="{self.log}"; REPO_URL="{repo_url}"; BRANCH=master\n'
            'STEP_NAME=test; API_KEY=""; PY=""; CHILD_PID=""; SPINNER_PID=""; SPINNER_STATUS=""\n'
            'init_ui; TTY=""\n'
            f'curl() {{ echo "$*" >> "{self.curl_calls}"; cat "{self.curl_code}"; }}\n')
        if self.answers:
            prelude += (
                f'exec 7< "{self.answers}"\n'
                f'ask() {{ echo "$1" >> "{self.asked}"; IFS= read -r "$2" <&7 || eval "$2=\'\'"; }}\n')
        run_env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
        run_env.update(env)
        return subprocess.run(['bash', '-c', prelude + script], env=run_env,
                              capture_output=True, text=True, timeout=30)

    @property
    def prompts(self) -> list[str]:
        return self.asked.read_text().splitlines() if self.asked.exists() else []


@pytest.fixture
def harness(tmp_path):
    return InstallShHarness(tmp_path)


PIP_RESOLVING = (
    'Collecting anthropic\n'
    '  Downloading anthropic-0.40.0-py3-none-any.whl.metadata (2 kB)\n'
    'Collecting numpy\n'
    '  Downloading numpy-2.1.0-cp312-cp312-macosx_14_0_arm64.whl.metadata (60 kB)\n')
PIP_DOWNLOADING = PIP_RESOLVING + (
    'Downloading anthropic-0.40.0-py3-none-any.whl (280 kB)\n'
    'Downloading numpy-2.1.0-cp312-cp312-macosx_14_0_arm64.whl (13.5 MB)\n')
PIP_BUILDING = PIP_DOWNLOADING + 'Building wheels for collected packages: pyaudio\n'
PIP_UNPACKING = PIP_BUILDING + 'Installing collected packages: anthropic, numpy, pyaudio\n'


class TestPipStatus:
    """pip only reveals the package total at the end, so the bar is scaled by an expected
    count and spans two passes: metadata resolution ("Collecting", one per package) and
    the wheel downloads (unindented "Downloading", one per package). The 40 s bug this
    guards against: counting one pass only, the bar sat at 100% for the whole download."""

    def raw(self, harness, log: str, expected: int, offset: int = 1) -> str:
        harness.log.write_text(log)
        result = harness.run(f'pip_status {offset} {expected}')
        assert result.returncode == 0, result.stderr
        return result.stdout

    def status(self, harness, log: str, expected: int, offset: int = 1) -> tuple[int, str]:
        out = self.raw(harness, log, expected, offset)
        match = BAR_RE.match(out)
        assert match, f"no progress bar in {out!r}"
        bar, text = match.groups()
        assert len(bar) == 24, f"bar is not 24 cells wide: {bar!r}"
        return bar.count('█'), text

    def test_resolving_pass_fills_the_first_half(self, harness):
        # 2 of 4 expected packages resolved, none downloaded: 2 of 8 units → 6 of 24 cells.
        # The indented metadata downloads of the resolve pass must not count as wheels.
        assert self.status(harness, PIP_RESOLVING, expected=4) == (6, 'resolving 2 packages')

    def test_download_pass_names_the_wheel_and_fills_the_second_half(self, harness):
        assert self.status(harness, PIP_DOWNLOADING, expected=4) == (12, 'downloading numpy 2.1.0 (13.5 MB)')

    def test_more_packages_than_expected_never_overflow_the_bar(self, harness):
        filled, _ = self.status(harness, PIP_DOWNLOADING, expected=1)
        assert filled == 24

    def test_build_and_unpack_passes_show_as_words(self, harness):
        # pip prints nothing per package in these passes, so a bar would only stand still.
        assert self.raw(harness, PIP_BUILDING, expected=4) == 'building source packages'
        assert self.raw(harness, PIP_UNPACKING, expected=4) == 'unpacking 3 packages'

    def test_offset_ignores_what_the_log_held_before_the_pip_step(self, harness):
        earlier = 'Collecting pip\nDownloading pip-24.2-py3-none-any.whl (1.8 MB)\n'
        _, text = self.status(harness, earlier + PIP_RESOLVING, expected=4, offset=len(earlier) + 1)
        assert text == 'resolving 2 packages'


class TestObtainApiKey:
    def test_environment_key_is_verified_without_a_prompt(self, harness):
        harness.set_answers()
        result = harness.run('obtain_api_key; printf "%s" "$API_KEY"', ANTHROPIC_API_KEY='sk-ant-env-47L')
        assert result.returncode == 0, result.stdout
        assert result.stdout.endswith('sk-ant-env-47L')
        assert 'Key verified' in result.stdout
        assert harness.prompts == []
        assert 'x-api-key: sk-ant-env-47L' in harness.curl_calls.read_text()

    def test_rejected_environment_key_stops_the_install(self, harness):
        harness.set_curl_code('401')
        result = harness.run('obtain_api_key', ANTHROPIC_API_KEY='sk-ant-env-47L')
        assert result.returncode == 1
        assert 'ANTHROPIC_API_KEY from the environment was rejected' in result.stdout

    def test_typed_key_is_trimmed_and_a_malformed_one_never_reaches_anthropic(self, harness):
        harness.set_answers('bogus-key', '  sk-ant-typed-47L  ')
        result = harness.run('obtain_api_key; printf "%s" "$API_KEY"')
        assert result.returncode == 0, result.stdout
        assert result.stdout.endswith('sk-ant-typed-47L')
        assert 'does not look like an Anthropic key' in result.stdout
        assert len(harness.prompts) == 2
        assert harness.curl_calls.read_text().count('x-api-key:') == 1
        assert 'bogus-key' not in harness.curl_calls.read_text()

    def test_three_rejected_keys_stop_the_install(self, harness):
        harness.set_curl_code('403')
        harness.set_answers('sk-ant-one', 'sk-ant-two', 'sk-ant-three', 'sk-ant-four')
        result = harness.run('obtain_api_key')
        assert result.returncode == 1
        assert 'No valid API key after 3 attempts' in result.stdout
        assert len(harness.prompts) == 3

    def test_unverifiable_key_is_kept_with_a_warning(self, harness):
        # Offline or an Anthropic outage must not block the install; the server reports a bad key.
        harness.set_curl_code('000')
        harness.set_answers('sk-ant-typed-47L')
        result = harness.run('obtain_api_key; printf "%s" "$API_KEY"')
        assert result.returncode == 0, result.stdout
        assert result.stdout.endswith('sk-ant-typed-47L')
        assert 'Could not verify the key right now' in result.stdout

    def test_no_terminal_and_no_environment_key_fails_with_the_workaround(self, harness):
        # curl | bash with no /dev/tty: `read` must not wait on the pipe the script came from.
        result = harness.run('obtain_api_key')
        assert result.returncode == 1
        assert 'No terminal to ask on' in result.stdout
        assert 'set the value in the environment' in result.stdout


class TestWriteConfig:
    def test_writes_a_private_config_with_every_required_key(self, harness):
        harness.keep_dir.mkdir()
        result = harness.run('API_KEY=sk-ant-typed-47L; write_config')
        assert result.returncode == 0, result.stdout
        config_path = harness.keep_dir / 'config.yaml'
        assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
        config = yaml.safe_load(config_path.read_text())
        for key in REQUIRED_KEYS:
            section, name = key.split('.')
            assert config[section][name], f"{key} is empty in the written config"
        assert config['api_keys']['anthropic'] == 'sk-ant-typed-47L'
        assert re.fullmatch(r'[A-Za-z0-9]{32}', str(config['vault']['key']))


def make_origin(path: Path) -> None:
    """A throwaway upstream on `master`, ignoring log/ like the real repository does — the
    installer's own log lives inside the checkout and must not count as a local change."""
    path.mkdir()
    (path / 'readme.md').write_text('Keep\n')
    (path / '.gitignore').write_text('log/\n')
    subprocess.run(['git', 'init', '-q', '-b', 'master'], cwd=path, check=True)
    commit(path, 'feat: first')


def commit(path: Path, message: str) -> str:
    subprocess.run(['git', *GIT_AUTHOR, 'add', '-A'], cwd=path, check=True)
    subprocess.run(['git', *GIT_AUTHOR, 'commit', '-qm', message], cwd=path, check=True)
    return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=path,
                          capture_output=True, text=True, check=True).stdout.strip()


class TestEnsureRepo:
    ENSURE = 'open_log; ensure_repo; printf "%s" "$LOG"'

    def test_fresh_install_clones_and_keeps_its_log_inside_the_checkout(self, harness):
        origin = harness.dir / 'origin'
        make_origin(origin)
        result = harness.run(self.ENSURE, repo_url=str(origin))
        assert result.returncode == 0, result.stdout
        assert (harness.keep_dir / '.git').is_dir()
        assert (harness.keep_dir / 'readme.md').read_text() == 'Keep\n'
        final_log = harness.keep_dir / 'log' / 'install.log'
        assert result.stdout.endswith(str(final_log))
        assert 'git clone' in final_log.read_text()
        assert not list(harness.dir.glob('keep-install*')), "the parked log was not moved back"

    def test_rerun_fast_forwards_an_existing_checkout(self, harness):
        origin = harness.dir / 'origin'
        make_origin(origin)
        assert harness.run(self.ENSURE, repo_url=str(origin)).returncode == 0
        (origin / 'readme.md').write_text('Keep, updated\n')
        newer = commit(origin, 'feat: second')
        result = harness.run(self.ENSURE, repo_url=str(origin))
        assert result.returncode == 0, result.stdout
        assert f'At {newer}' in result.stdout
        assert (harness.keep_dir / 'readme.md').read_text() == 'Keep, updated\n'
        assert 'Local changes' not in result.stdout, "the installer's own log counted as a local change"

    def test_checkout_of_another_repository_is_refused(self, harness):
        origin = harness.dir / 'origin'
        make_origin(origin)
        assert harness.run(self.ENSURE, repo_url=str(origin)).returncode == 0
        result = harness.run(self.ENSURE, repo_url='https://git.example/alice/Other.git')
        assert result.returncode == 1
        assert 'is a git checkout of' in result.stdout
        assert 'KEEP_DIR=' in result.stdout

    def test_non_empty_directory_is_refused_and_left_untouched(self, harness):
        harness.keep_dir.mkdir()
        (harness.keep_dir / 'notes.txt').write_text('mine\n')
        result = harness.run(self.ENSURE, repo_url=str(harness.dir / 'origin'))
        assert result.returncode == 1
        assert 'is not a Keep checkout' in result.stdout
        assert sorted(p.name for p in harness.keep_dir.iterdir()) == ['notes.txt']
        parked = re.search(r'Log: (\S+)', result.stdout).group(1)
        assert not parked.startswith(str(harness.keep_dir))
        Path(parked).unlink()
