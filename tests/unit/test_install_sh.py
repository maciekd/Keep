"""Tests for install.sh's decision logic: the pip progress read off pip's own log, the
API key intake, the config.yaml it writes and the state machine around the checkout
(fresh clone, upgrade, refused directories).

The script's functions are sourced without the trailing `main "$@"` call and driven
one at a time. Stdout is a pipe, so the spinner (a background subshell) is off and
the output is plain text. `curl` and the terminal prompt `ask` are bash functions
sourced before the call (see test_start_sh.py for why functions, not executables);
the repository tests run real git against a throwaway origin in tmp_path.

Either system's branch runs on either system: `KEEP_OS` names it, and the harness pins the Mac's
unless a test asks for Linux. Whatever the Linux branch would do to the machine — `sudo`,
`apt-get`, `dpkg-query`, `apt-cache`, `systemctl`, `loginctl` — is a function of the harness
answering from files in tmp_path, sourced before every call whichever system is asked for: no
test reaches the real one.
"""
import json
import os
import re
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from common.ConfigManager import REQUIRED_KEYS

from .git_sandbox import History, empty_repo, git, git_env, spawn
from .test_start_sh import UNIT_SAFE_NAMES, UNIT_UNSAFE_NAMES
from .test_role_sh import REFUSED_CONFIGS, ROLE_SH, SATELLITE_CONFIGS, SERVER_CONFIGS, UNREADABLE_CONFIGS, python_reading

INSTALL_SH = Path(__file__).resolve().parents[2] / 'install.sh'
WORKSPACE_REQUIRED_BODY = (
    '{"type":"error","error":{"type":"invalid_request_error","message":"anthropic-workspace-id is required '
    'when authenticating with an identity-linked API key; send the id of the workspace this request acts in."},'
    '"request_id":"req_test_0000000000000000"}')
CREDIT_TOO_LOW_BODY = (
    '{"type":"error","error":{"type":"invalid_request_error",'
    '"message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing '
    'to upgrade or purchase credits."}}')
BAR_RE = re.compile(r'^\[(.*)\] (.*)$')
MAC, LINUX = 'Darwin', 'Linux'      # as `uname -s` names them, and KEEP_OS
UBUNTU = 'PRETTY_NAME="Ubuntu 24.04.1 LTS"\nNAME="Ubuntu"\nVERSION_ID="24.04"\nID=ubuntu\nID_LIKE=debian\n'
DEBIAN = 'PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"\nNAME="Debian GNU/Linux"\nVERSION_ID="12"\nID=debian\n'
MINT = 'PRETTY_NAME="Linux Mint 22"\nVERSION_ID="22"\nID=linuxmint\nID_LIKE="ubuntu debian"\n'
FEDORA = 'PRETTY_NAME="Fedora Linux 41 (Server Edition)"\nVERSION_ID=41\nID=fedora\n'
BROWSERS = ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser']


def machine_stubs(d: Path) -> str:
    """The Linux machine as bash functions. `installed` lists the packages dpkg has; `sudo` runs
    nothing: an `apt-get install` handed to it adds the named packages to that list. A command
    named in `absent` is not found by `command -v`, whatever the real machine has."""
    return f"""
command() {{
    if [ "$1" = -v ] && grep -qx -- "$2" "{d}/absent" 2>/dev/null; then return 1; fi
    builtin command "$@"
}}
sudo() {{
    echo "$*" >> "{d}/sudo_calls"
    case " $* " in *" -v "*) [ ! -f "{d}/sudo_needs_password" ]; return ;; esac
    if [ -f "{d}/apt_fails" ]; then echo "E: Unable to locate package"; return 100; fi
    if [ -f "{d}/apt_broken" ]; then echo "E: dpkg was interrupted"; return 100; fi
    case " $* " in *" update "*) rm -f "{d}/apt_lists_empty"; return ;; esac
    if [ -f "{d}/apt_lists_empty" ]; then echo "E: Unable to locate package"; return 100; fi
    local a installing=""
    for a in "$@"; do
        if [ -n "$installing" ]; then case "$a" in -*) ;; *) echo "$a" >> "{d}/installed" ;; esac; fi
        [ "$a" != install ] || installing=1
    done
}}
apt-get() {{ echo "$*" >> "{d}/apt_get_calls"; return 1; }}
dpkg-query() {{
    local name; for name in "$@"; do :; done
    grep -qx -- "$name" "{d}/installed" 2>/dev/null && printf installed
}}
apt-cache() {{
    if [ -f "{d}/apt_offers_python" ] && [ ! -f "{d}/apt_lists_empty" ]; then echo "  Candidate: 3.12.3-1ubuntu0.5"; else echo "  Candidate: (none)"; fi
}}
systemctl() {{
    echo "$*" >> "{d}/systemctl_calls"
    [ ! -f "{d}/no_user_systemd" ] || {{ echo "Failed to connect to bus: No medium found" >&2; return 1; }}
}}
loginctl() {{ echo "$*" >> "{d}/loginctl_calls"; [ ! -f "{d}/linger_refused" ]; }}
"""


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
        self.asked = tmp_path / 'asked'
        self.answers = None
        self.os_release = tmp_path / 'os-release'
        self.os_release.write_text(UBUNTU)
        self.set_curl(('200', ''))

    def flag(self, name: str):
        """One of the files the machine's stubs answer by: `no_user_systemd`, `apt_fails`, …"""
        (self.dir / name).write_text('')

    def installed(self, *packages: str):
        (self.dir / 'installed').write_text(''.join(f'{name}\n' for name in packages))

    def absent(self, *commands: str):
        """Commands `command -v` does not find."""
        with (self.dir / 'absent').open('a') as f:
            f.write(''.join(f'{name}\n' for name in commands))

    def calls(self, tool: str) -> list[str]:
        path = self.dir / f'{tool}_calls'
        return path.read_text().splitlines() if path.exists() else []

    @property
    def apt_installs(self) -> list[list[str]]:
        """The packages of every `apt-get install` handed to sudo."""
        return [call.split('--no-install-recommends ', 1)[1].split() for call in self.calls('sudo') if ' install ' in call]

    def set_curl(self, *responses: tuple[str, str]):
        """What successive curl calls answer, as (status, body) pairs; the last one repeats."""
        for n, (code, body) in enumerate(responses, 1):
            (self.dir / f'curl_{n}.code').write_text(code)
            (self.dir / f'curl_{n}.body').write_text(body)
        (self.dir / 'curl_count').write_text(str(len(responses)))

    def set_curl_code(self, code: str):
        self.set_curl((code, ''))

    def set_answers(self, *lines: str):
        """What the user types at successive prompts; the prompt is recorded in `asked`."""
        self.answers = self.dir / 'answers'
        self.answers.write_text(''.join(f'{line}\n' for line in lines))

    def run(self, script: str, repo_url: str = '', no_terminal: bool = False, **env) -> subprocess.CompletedProcess:
        """`no_terminal`: in a session of its own, so the script has no controlling terminal
        whatever pytest was started from — ssh without -t, launchd."""
        prelude = (
            f'source "{self.fns}"\n'
            f'KEEP_DIR="{self.keep_dir}"; LOG="{self.log}"; REPO_URL="{repo_url}"; BRANCH=master\n'
            'STEP_NAME=test; API_KEY=""; WORKSPACE_ID=""; VERIFY_CODE=""; ANTHROPIC_SAID=""\n'
            'PY=""; CHILD_PID=""; SPINNER_PID=""; SPINNER_STATUS=""\n'
            'init_ui; TTY=""\n'
            f'OS_RELEASE="{self.os_release}"; CPUINFO="{self.dir}/cpuinfo"\n'
            + machine_stubs(self.dir) +
            # The n-th call answers with curl_<n>.code and, for its -o target, curl_<n>.body.
            f'curl() {{ echo "$*" >> "{self.curl_calls}"; local n prev="" a\n'
            f'  n=$(( $(cat "{self.dir}/curl_n" 2>/dev/null || echo 0) + 1 )); echo "$n" > "{self.dir}/curl_n"\n'
            f'  [ "$n" -le "$(cat "{self.dir}/curl_count")" ] || n=$(cat "{self.dir}/curl_count")\n'
            f'  for a in "$@"; do [ "$prev" = -o ] && cat "{self.dir}/curl_$n.body" > "$a"; prev=$a; done\n'
            f'  cat "{self.dir}/curl_$n.code"; }}\n')
        if self.answers:
            prelude += (
                f'exec 7< "{self.answers}"\n'
                f'ask() {{ echo "$1" >> "{self.asked}"; IFS= read -r "$2" <&7 || eval "$2=\'\'"; }}\n')
        run_env = {k: v for k, v in git_env().items() if k != 'ANTHROPIC_API_KEY'}
        run_env['KEEP_OS'] = MAC
        run_env.update(env)
        return spawn(['bash', '-c', prelude + script], env=run_env, timeout=30, start_new_session=no_terminal)

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

    def test_an_empty_enter_and_a_slip_of_the_paste_use_up_no_attempt(self, harness):
        """Three attempts are three keys Anthropic was asked about."""
        harness.set_curl(('403', ''), ('403', ''), ('200', ''))
        harness.set_answers('', 'sk-ant-one', 'oops', '', 'sk-ant-two', '', 'bogus', 'sk-ant-three')
        result = harness.run('obtain_api_key; printf "%s" "$API_KEY"')
        assert result.returncode == 0, result.stdout
        assert result.stdout.endswith('sk-ant-three')
        assert harness.curl_calls.read_text().count('x-api-key:') == 3

    def test_enter_pressed_for_ever_ends_the_install(self, harness):
        harness.set_answers('')
        result = harness.run('obtain_api_key')
        assert result.returncode == 1
        assert not harness.curl_calls.exists() or 'x-api-key:' not in harness.curl_calls.read_text()

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

    def test_key_without_a_workspace_asks_for_the_id_and_sends_it_from_then_on(self, harness):
        """The 2026-10 beta tester: the Console's Create key dialog left unscoped makes a key
        Anthropic refuses without an anthropic-workspace-id header. The check is a real
        message so this 400 shows at install, not at the first chat."""
        harness.set_curl(('400', WORKSPACE_REQUIRED_BODY), ('200', ''))
        harness.set_answers('sk-ant-typed-47L', 'wrkspc_01TEST')
        result = harness.run('obtain_api_key; printf "%s %s" "$API_KEY" "$WORKSPACE_ID"')
        assert result.returncode == 0, result.stdout
        assert result.stdout.endswith('sk-ant-typed-47L wrkspc_01TEST')
        assert 'Scope' in result.stdout and 'Organization' in result.stdout   # names the dialog's own field
        assert 'platform.claude.com/settings/keys' in result.stdout
        assert 'platform.claude.com/settings/workspaces' in result.stdout
        assert len(harness.prompts) == 2 and 'Workspace id' in harness.prompts[1]
        first, second = harness.curl_calls.read_text().splitlines()
        assert 'anthropic-workspace-id' not in first
        assert 'anthropic-workspace-id: wrkspc_01TEST' in second

    def test_enter_instead_of_a_workspace_id_asks_for_a_new_key(self, harness):
        harness.set_curl(('400', WORKSPACE_REQUIRED_BODY), ('200', ''))
        harness.set_answers('sk-ant-one', '', 'sk-ant-two')
        result = harness.run('obtain_api_key; printf "%s %s" "$API_KEY" "$WORKSPACE_ID"')
        assert result.returncode == 0, result.stdout
        assert result.stdout.endswith('sk-ant-two ')
        assert len(harness.prompts) == 3
        assert 'anthropic-workspace-id' not in harness.curl_calls.read_text()

    def test_unknown_workspace_id_is_asked_again(self, harness):
        not_found = '{"type":"error","error":{"type":"not_found_error","message":"Workspace `wrkspc_bad` not found."}}'
        harness.set_curl(('400', WORKSPACE_REQUIRED_BODY), ('404', not_found), ('200', ''))
        harness.set_answers('sk-ant-one', 'wrkspc_bad', 'wrkspc_good')
        result = harness.run('obtain_api_key; printf "%s" "$WORKSPACE_ID"')
        assert result.returncode == 0, result.stdout
        assert result.stdout.endswith('wrkspc_good')
        assert 'does not know workspace wrkspc_bad' in result.stdout
        assert len(harness.prompts) == 3

    def test_malformed_workspace_id_never_reaches_anthropic(self, harness):
        harness.set_curl(('400', WORKSPACE_REQUIRED_BODY), ('200', ''))
        harness.set_answers('sk-ant-one', 'default', 'wrkspc_01TEST')
        result = harness.run('obtain_api_key')
        assert result.returncode == 0, result.stdout
        assert 'does not look like a workspace id' in result.stdout
        assert 'default' not in harness.curl_calls.read_text()

    def test_environment_workspace_id_is_sent_without_a_prompt(self, harness):
        harness.set_answers()
        result = harness.run('obtain_api_key', ANTHROPIC_API_KEY='sk-ant-env-47L',
                             ANTHROPIC_WORKSPACE_ID='wrkspc_01ENV')
        assert result.returncode == 0, result.stdout
        assert harness.prompts == []
        assert 'anthropic-workspace-id: wrkspc_01ENV' in harness.curl_calls.read_text()
        assert 'wrkspc_01ENV' in result.stdout

    def test_environment_key_without_a_workspace_names_the_variable_to_set(self, harness):
        harness.set_curl(('400', WORKSPACE_REQUIRED_BODY))
        harness.set_answers()
        result = harness.run('obtain_api_key', ANTHROPIC_API_KEY='sk-ant-env-47L')
        assert result.returncode == 1
        assert 'ANTHROPIC_WORKSPACE_ID' in result.stdout

    def test_other_400_keeps_the_key_and_repeats_what_anthropic_said(self, harness):
        other = ('{"type":"error","error":{"type":"invalid_request_error",'
                 '"message":"This organization has been disabled."}}')
        harness.set_curl(('400', other))
        harness.set_answers('sk-ant-typed-47L')
        result = harness.run('obtain_api_key; printf "%s" "$API_KEY"')
        assert result.returncode == 0, result.stdout
        assert result.stdout.endswith('sk-ant-typed-47L')
        assert 'Could not verify the key right now' in result.stdout
        assert 'organization has been disabled' in result.stdout

    def test_check_is_a_message_so_an_unfunded_organization_shows_at_install(self, harness):
        """The free count_tokens endpoint is not documented to check the balance: a key whose
        organization has no credit could pass it and leave the first chat silent."""
        harness.set_answers('sk-ant-typed-47L')
        harness.run('obtain_api_key')
        assert harness.curl_calls.read_text().split()[-1] == 'https://api.anthropic.com/v1/messages'

    def test_unfunded_organization_waits_for_credit_and_checks_the_same_key_again(self, harness):
        harness.set_curl(('400', CREDIT_TOO_LOW_BODY), ('200', ''))
        harness.set_answers('sk-ant-typed-47L', '')
        result = harness.run('obtain_api_key; printf "%s" "$API_KEY"')
        assert result.returncode == 0, result.stdout
        assert result.stdout.endswith('sk-ant-typed-47L')
        assert 'Key verified' in result.stdout
        assert 'platform.claude.com/settings/billing' in result.stdout
        assert len(harness.prompts) == 2 and 'API key' not in harness.prompts[1]
        assert harness.curl_calls.read_text().count('x-api-key: sk-ant-typed-47L') == 2

    def test_skip_keeps_a_key_whose_organization_has_no_credit(self, harness):
        # A new Max subscriber waits 7 days for the plan's credit: the install must still finish.
        harness.set_curl(('400', CREDIT_TOO_LOW_BODY))
        harness.set_answers('sk-ant-typed-47L', ' SKIP ')
        result = harness.run('obtain_api_key; printf "%s" "$API_KEY"')
        assert result.returncode == 0, result.stdout
        assert result.stdout.endswith('sk-ant-typed-47L')
        assert harness.curl_calls.read_text().count('x-api-key:') == 1

    def test_environment_key_without_credit_is_kept_without_a_prompt(self, harness):
        harness.set_curl(('400', CREDIT_TOO_LOW_BODY))
        harness.set_answers()
        result = harness.run('obtain_api_key; printf "%s" "$API_KEY"', ANTHROPIC_API_KEY='sk-ant-env-47L')
        assert result.returncode == 0, result.stdout
        assert result.stdout.endswith('sk-ant-env-47L')
        assert harness.prompts == []

    def test_no_terminal_and_no_environment_key_fails_with_the_workaround(self, harness):
        # curl | bash with no /dev/tty: `read` must not wait on the pipe the script came from.
        result = harness.run('obtain_api_key')
        assert result.returncode == 1
        assert 'No terminal to ask on' in result.stdout
        assert 'set the value in the environment' in result.stdout


class TestWhetherThereIsATerminal:
    """`ssh mini 'curl ... | bash'`: /dev/tty is there, readable and writable, and cannot be
    opened. Taking it for a terminal made the first question the end of the install."""

    SHOW = 'init_ui; printf "TTY=[%s]" "$TTY"'

    def test_a_process_without_a_controlling_terminal_has_none_to_ask_on(self, harness):
        result = harness.run(self.SHOW, no_terminal=True)
        assert result.returncode == 0, result.stdout
        assert result.stdout.endswith('TTY=[]')

    def test_a_terminal_that_can_be_opened_is_the_one_to_ask_on(self, harness):
        """`pty.spawn` gives the installer a pseudo-terminal of its own as the controlling one —
        from a Python of its own, so that the worker, which has threads, does not fork."""
        command = f'source "{harness.fns}"; {self.SHOW}'
        result = subprocess.run([sys.executable, '-c', 'import pty, sys; pty.spawn(sys.argv[1:])', '/bin/bash', '-c', command],
                                capture_output=True, text=True, stdin=subprocess.DEVNULL, start_new_session=True, timeout=30)
        assert result.stdout.endswith('TTY=[/dev/tty]'), result.stdout + result.stderr

    def test_a_satellite_installs_without_a_terminal_and_answers_this_mac_only(self, harness):
        main_on_the_real_terminal_check = MAIN.replace('init_ui() { real_init_ui; TTY=""; }\n', '')
        result = harness.run(main_on_the_real_terminal_check, no_terminal=True, KEEP_ROLE=SATELLITE, KEEP_NO_TELEMETRY='1')
        assert result.returncode == 0, result.stdout
        config = yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())
        assert config['imessage_satellite']['lan_access'] is False

    def test_a_server_install_without_a_terminal_names_the_way_out_instead_of_dying_on_its_question(self, harness):
        result = harness.run('trap on_error ERR; init_ui; obtain_api_key', no_terminal=True)
        assert result.returncode == 1
        assert 'ANTHROPIC_API_KEY' in result.stdout or 'environment' in result.stdout
        assert 'Unexpected error' not in result.stdout


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
        assert 'anthropic_workspace' not in config['api_keys']   # a key tied to one workspace needs none
        assert re.fullmatch(r'[A-Za-z0-9]{32}', str(config['vault']['key']))

    def test_the_config_has_no_empty_line_where_the_workspace_id_would_stand(self, harness):
        harness.keep_dir.mkdir(parents=True, exist_ok=True)
        result = harness.run('API_KEY=sk-ant-typed-47L; write_config')
        assert result.returncode == 0, result.stdout
        lines = (harness.keep_dir / 'config.yaml').read_text().splitlines()
        assert lines[lines.index('  anthropic: sk-ant-typed-47L') + 1].startswith('developer_release:')

    def test_workspace_id_is_written_for_a_key_not_tied_to_one(self, harness):
        """Contract with config.yaml: ClaudeLLM reads the id as api_keys.anthropic_workspace."""
        harness.keep_dir.mkdir()
        result = harness.run('API_KEY=sk-ant-typed-47L; WORKSPACE_ID=wrkspc_01TEST; write_config')
        assert result.returncode == 0, result.stdout
        config = yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())
        assert config['api_keys']['anthropic_workspace'] == 'wrkspc_01TEST'

    def test_developer_release_follows_the_branch_the_install_was_made_from(self, harness):
        """restart_server pulls the branch `developer_release` names, so a master install
        must keep following master and a release install the release branch."""
        harness.keep_dir.mkdir()
        assert harness.run('API_KEY=sk-ant-typed-47L; write_config').returncode == 0
        config = yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())
        assert config['developer_release'] is True
        assert harness.run('BRANCH=release; API_KEY=sk-ant-typed-47L; write_config').returncode == 0
        config = yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())
        assert config['developer_release'] is False


FIRST = {'readme.md': 'Keep\n', '.gitignore': 'log/\n'}


def make_origin(path: Path) -> History:
    """A throwaway upstream on `master`, ignoring log/ like the real repository does — the
    installer's own log lives inside the checkout and must not count as a local change."""
    history = History()
    history.commit(datetime(2026, 9, 1, 12, tzinfo=timezone.utc), 'feat: first', FIRST)
    history.write(empty_repo(path, bare=True))
    return history


def move_on(origin: Path, history: History, readme: str) -> str:
    """A second commit on origin's master; returns its short hash."""
    history.commit(datetime(2026, 9, 2, 12, tzinfo=timezone.utc), 'feat: second', {**FIRST, 'readme.md': readme})
    return history.write(origin)[-1][:7]


def install(keep_dir: Path, origin: Path, history: History):
    """The checkout an earlier install left in `keep_dir`: origin as it is now, on master."""
    history.write(empty_repo(keep_dir, origin=origin))
    git(keep_dir, 'reset', '-q', '--hard')


def branch_of(path: Path) -> str:
    return (path / '.git' / 'HEAD').read_text().strip().removeprefix('ref: refs/heads/')


class TestEnsureRepo:
    """The origin and an existing checkout come from git_sandbox, written without git
    processes; the script's own git calls are what a test has time for."""

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
        history = make_origin(origin)
        install(harness.keep_dir, origin, history)
        newer = move_on(origin, history, 'Keep, updated\n')
        result = harness.run(self.ENSURE, repo_url=str(origin))
        assert result.returncode == 0, result.stdout
        assert f'At {newer}' in result.stdout
        assert (harness.keep_dir / 'readme.md').read_text() == 'Keep, updated\n'
        assert 'Local changes' not in result.stdout, "the installer's own log counted as a local change"

    def test_rerun_keeps_the_branch_an_existing_checkout_is_on(self, harness):
        """The default branch became `release`, but a checkout made from master (the
        development machine, an early install) must not be moved by a re-run."""
        origin = harness.dir / 'origin'
        history = make_origin(origin)
        install(harness.keep_dir, origin, history)
        history.branch('release', 1)
        newer = move_on(origin, history, 'Keep, master moved on\n')
        result = harness.run('BRANCH=release; ' + self.ENSURE, repo_url=str(origin))
        assert result.returncode == 0, result.stdout
        assert f'At {newer}' in result.stdout
        assert branch_of(harness.keep_dir) == 'master'

    def test_checkout_of_another_repository_is_refused(self, harness):
        empty_repo(harness.keep_dir, origin=harness.dir / 'origin')
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

    def test_run_that_failed_before_the_clone_resumes_into_the_same_log(self, harness):
        """The first run creates log/install.log before it can fail (no network, an
        interrupted CLT wait); the re-run used to die in git clone on the non-empty dir."""
        origin = harness.dir / 'origin'
        make_origin(origin)
        first = harness.run('open_log; log "network failed"; printf "%s" "$LOG"')
        assert first.stdout.endswith(str(harness.keep_dir / 'log' / 'install.log'))
        result = harness.run(self.ENSURE, repo_url=str(origin))
        assert result.returncode == 0, result.stdout
        final_log = (harness.keep_dir / 'log' / 'install.log').read_text()
        assert 'network failed' in final_log and 'git clone' in final_log
        assert not list(harness.dir.glob('keep-install*'))

    def test_foreign_file_in_the_log_directory_still_refuses(self, harness):
        (harness.keep_dir / 'log').mkdir(parents=True)
        (harness.keep_dir / 'log' / 'install.log').write_text('earlier run\n')
        (harness.keep_dir / 'log' / 'floxy.out').write_text('not ours\n')
        result = harness.run(self.ENSURE, repo_url=str(harness.dir / 'origin'))
        assert result.returncode == 1
        assert 'is not a Keep checkout' in result.stdout
        assert (harness.keep_dir / 'log' / 'floxy.out').read_text() == 'not ours\n'



class TestTelemetry:
    """The install reports to PostHog: the id it mints is the one the server's events carry
    from then on, a re-run reads the id and the opt-out from the existing config.yaml, and
    an event is one curl that can never stop the install."""

    def event(self, harness) -> dict:
        lines = harness.curl_calls.read_text().splitlines()
        assert len(lines) == 1, lines
        match = re.search(r'-d (\{.*\}) https://eu\.i\.posthog\.com/capture/$', lines[0])
        assert match, lines[0]
        return json.loads(match.group(1))

    def test_project_key_and_host_are_the_servers(self):
        """Contract with PostHog: install.sh runs before the clone, so it carries its own copy
        of the key and host Telemetry.py posts to; both must stay the same project."""
        from assistant import Telemetry
        text = INSTALL_SH.read_text()
        assert re.search(r'^POSTHOG_PROJECT_KEY="(.*)"$', text, re.M).group(1) == Telemetry.POSTHOG_PROJECT_KEY
        assert re.search(r'^POSTHOG_HOST="(.*)"$', text, re.M).group(1) == Telemetry.POSTHOG_HOST

    def test_fresh_install_mints_an_id_and_writes_it_as_the_servers(self, harness):
        """Contract with config.yaml: the server reads the install id as telemetry.id."""
        harness.keep_dir.mkdir()
        result = harness.run('init_telemetry; echo "id=$TELEMETRY_ID"; API_KEY=sk-ant-typed-47L; write_config')
        assert result.returncode == 0, result.stdout
        minted = re.search(r'^id=(.*)$', result.stdout, re.M).group(1)
        assert re.fullmatch(r'[0-9a-f-]{36}', minted), minted
        config = yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())
        assert config['telemetry'] == {'enabled': True, 'id': minted}

    def test_rerun_reuses_the_id_the_config_holds(self, harness):
        harness.keep_dir.mkdir()
        (harness.keep_dir / 'config.yaml').write_text(
            'api_keys:\n  anthropic: sk-ant-typed-47L\ntelemetry:\n  enabled: true   # on\n  id: "11111111-2222-4333-8444-555555555555"\nvault:\n  key: x\n')
        result = harness.run('init_telemetry; echo "id=$TELEMETRY_ID"')
        assert 'id=11111111-2222-4333-8444-555555555555' in result.stdout

    def test_opt_out_in_the_config_sends_nothing(self, harness):
        harness.keep_dir.mkdir()
        (harness.keep_dir / 'config.yaml').write_text(
            'telemetry:\n  enabled: false\n  id: 11111111-2222-4333-8444-555555555555\n')
        result = harness.run('init_telemetry; telemetry_event install_started fresh=false; echo "id=[$TELEMETRY_ID]"')
        assert 'id=[]' in result.stdout
        assert not harness.curl_calls.exists()

    def test_environment_opt_out_sends_nothing_and_writes_the_opt_out(self, harness):
        harness.keep_dir.mkdir()
        result = harness.run('init_telemetry; telemetry_started; API_KEY=sk-ant-typed-47L; write_config',
                             KEEP_NO_TELEMETRY='1')
        assert result.returncode == 0, result.stdout
        assert not harness.curl_calls.exists()
        config = yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())
        assert config['telemetry'] == {'enabled': False}

    def test_event_carries_the_id_and_typed_properties_without_the_home_path(self, harness):
        result = harness.run(
            'TELEMETRY_ID=11111111-2222-4333-8444-555555555555; HOME=/Users/tester\n'
            'telemetry_event install_finished elapsed_s=42 fresh=true '
            'commit=d4008b4 reason="Python at \'/Users/tester/py\' said \\"no\\""')
        assert result.returncode == 0, result.stdout
        event = self.event(harness)
        assert event['event'] == 'install_finished'
        assert event['distinct_id'] == '11111111-2222-4333-8444-555555555555'
        assert event['properties']['elapsed_s'] == 42
        assert event['properties']['fresh'] is True
        assert event['properties']['commit'] == 'd4008b4'
        assert event['properties']['reason'] == 'Python at \'~/py\' said "no"'
        assert '/Users/tester' not in harness.curl_calls.read_text()

    def test_failure_reports_the_step_and_still_exits_with_the_error(self, harness):
        result = harness.run('TELEMETRY_ID=11111111-2222-4333-8444-555555555555; STEP_NAME="Python 3.12"\n'
                             'fail "uv did not install." "Try again."')
        assert result.returncode == 1
        event = self.event(harness)
        assert event['event'] == 'install_failed'
        assert event['properties']['step'] == 'Python 3.12'
        assert event['properties']['reason'] == 'uv did not install.'

    def test_unreachable_posthog_never_stops_the_install(self, harness):
        harness.set_curl_code('000')
        result = harness.run('TELEMETRY_ID=11111111-2222-4333-8444-555555555555\n'
                             'telemetry_event install_started fresh=true; echo survived')
        assert result.returncode == 0
        assert 'survived' in result.stdout


# main() with every step that touches the Mac or the network replaced by a line saying it ran;
# the role, the config, the telemetry, the step counter and the summary are the real ones. The
# terminal is taken away, so nothing here can wait on a prompt.
MAIN = (
    'for step in check_mac ensure_clt ensure_python ensure_repo ensure_venv install_requirements obtain_api_key launch; do\n'
    '  eval "$step() { echo \\"ran:$step\\"; }"\n'
    'done\n'
    'eval "real_$(declare -f init_ui)"\n'
    'init_ui() { real_init_ui; TTY=""; }\n'
    'main\n')
SATELLITE = 'imessage_satellite'
EXISTING_SATELLITE = ('role: imessage_satellite\nimessage_satellite:\n  port: 8901\n'
                      '  token: Zq4nB8cR2yH6sD0fG3jA7vK1mX5pL9wT\n  lan_access: false\ntelemetry:\n  enabled: false\n')
EXISTING_SERVER = 'api_keys:\n  anthropic: sk-ant-typed-47L\nvault:\n  key: x\ntelemetry:\n  enabled: false\n'


def ran(result) -> list[str]:
    return re.findall(r'^ran:(\w+)$', result.stdout, re.M)


def steps(result) -> list[tuple[int, int]]:
    return [(int(n), int(total)) for n, total in re.findall(r'\[(\d+)/(\d+)\]', result.stdout)]


def loaded(path: Path):
    """A ConfigManager on the file, as the process started on that config has it."""
    import os
    from unittest.mock import patch

    from common.ConfigManager import ConfigManager
    with patch.dict(os.environ, FLOXY_CONFIG=str(path)):
        return ConfigManager()


def existing_install(harness, config: str):
    """A checkout an earlier install left, with its config.yaml."""
    (harness.keep_dir / '.git').mkdir(parents=True)
    (harness.keep_dir / 'config.yaml').write_text(config)


class TestSatelliteInstall:
    """KEEP_ROLE=imessage_satellite: the same installer makes an install that runs no assistant —
    no API key, a small config with a generated token, and a summary with what the Keep server
    needs to pair. An install without the switch is the server it always was."""

    def test_the_switch_takes_the_value_of_the_config_key(self, harness):
        """A contract with config.yaml's `role` key (common/Role.py): one spelling for both."""
        from common import Role
        result = harness.run('resolve_role; printf "%s" "$ROLE"', KEEP_ROLE=Role.IMESSAGE_SATELLITE)
        assert result.returncode == 0, result.stdout
        assert result.stdout.endswith(Role.IMESSAGE_SATELLITE)

    @pytest.mark.parametrize('config', [*SERVER_CONFIGS.values(), *SATELLITE_CONFIGS.values(), *REFUSED_CONFIGS.values()],
                             ids=[*SERVER_CONFIGS, *SATELLITE_CONFIGS, *REFUSED_CONFIGS])
    def test_an_existing_installs_role_is_read_as_python_reads_it(self, harness, config):
        """The installer reads config.yaml with its own awk (it runs before the checkout exists):
        the verdict must be Role.current()'s, or an upgrade would treat a satellite as a server."""
        from common import Role
        role, _ = python_reading(harness.dir, config)
        existing_install(harness, config)
        result = harness.run('resolve_role; printf "role=%s" "$ROLE"')
        if role in Role.ROLES:
            assert result.stdout.endswith(f'role={role}'), result.stdout
        else:
            assert result.returncode == 1 and 'role' in result.stdout

    @pytest.mark.parametrize('config', UNREADABLE_CONFIGS.values(), ids=UNREADABLE_CONFIGS.keys())
    def test_an_existing_install_whose_role_cannot_be_read_is_refused_not_taken_for_a_server(self, harness, config):
        """What Python starts as a satellite and the installer's reader does not follow: an
        upgrade that guessed "server" would ask for an API key and write a server's summary."""
        existing_install(harness, config)
        result = harness.run(MAIN, KEEP_NO_TELEMETRY='1')
        assert result.returncode == 1
        assert ran(result) == [], "the install went on with a role it could not read"
        assert (harness.keep_dir / 'config.yaml').read_bytes() == config.encode()

    def test_the_installer_reads_config_yaml_with_the_program_the_start_scripts_use(self):
        """install.sh runs before a checkout exists and cannot source scripts/role.sh: it carries
        the reader. Two readers that drift apart are an install start.sh takes for another role."""
        def reader(script: Path) -> str:
            source = script.read_text()
            return source[source.index('# >>> config reader'):source.index('# <<< config reader')]
        assert reader(INSTALL_SH) == reader(ROLE_SH)

    @pytest.mark.parametrize('value', ['imessage', 'satellite', 'imessage-satellite', '1'])
    def test_any_other_value_stops_the_install_before_it_starts(self, harness, value):
        result = harness.run(MAIN, KEEP_ROLE=value, KEEP_NO_TELEMETRY='1')
        assert result.returncode == 1
        assert 'KEEP_ROLE' in result.stdout
        assert ran(result) == []

    def test_a_fresh_satellite_install_skips_the_api_key_and_counts_its_own_steps(self, harness):
        result = harness.run(MAIN, KEEP_ROLE=SATELLITE, KEEP_NO_TELEMETRY='1')
        assert result.returncode == 0, result.stdout
        assert ran(result) == ['check_mac', 'ensure_clt', 'ensure_python', 'ensure_repo', 'ensure_venv',
                               'install_requirements', 'launch']
        counted = steps(result)
        assert counted == [(n, len(counted)) for n in range(1, len(counted) + 1)], "the step counter skips or overruns"
        assert 'API key' not in result.stdout

    def test_an_install_without_the_switch_is_still_a_server_that_asks_for_its_key(self, harness):
        result = harness.run(MAIN, KEEP_NO_TELEMETRY='1')
        assert result.returncode == 0, result.stdout
        assert 'obtain_api_key' in ran(result)
        counted = steps(result)
        assert counted == [(n, len(counted)) for n in range(1, len(counted) + 1)]
        config = yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())
        assert 'role' not in config and 'imessage_satellite' not in config and 'watchdog' not in config
        assert config['vault']['key']
        assert 'imessage.host' not in result.stdout

    def test_the_config_it_writes_is_one_the_satellite_starts_on(self, harness):
        """Contract with floxy_ai_server.py: the role it reads, and a section Satellite.py accepts."""
        from unittest.mock import patch

        from common import Role
        from common.imessage import Satellite
        result = harness.run(MAIN, KEEP_ROLE=SATELLITE, KEEP_NO_TELEMETRY='1')
        assert result.returncode == 0, result.stdout
        path = harness.keep_dir / 'config.yaml'
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        manager = loaded(path)
        with patch.object(Role, 'config', manager), patch.object(Satellite, 'config', manager):
            assert Role.current() == Role.IMESSAGE_SATELLITE
            assert Satellite.config_problem() == ''
            assert Satellite._port() == Satellite.DEFAULT_PORT
            assert Satellite.bind_host() == Satellite.LOOPBACK, "with nobody asked, the satellite opens to this Mac only"
        config = yaml.safe_load(path.read_text())
        assert re.fullmatch(r'[A-Za-z0-9]{32}', str(config['imessage_satellite']['token']))
        assert 'api_keys' not in config and 'vault' not in config

    def test_the_watchdog_is_on_for_a_satellite(self, harness):
        """Contract with common/Watchdog.KEY, which the satellite applies at each start: nobody is
        at a satellite to start it again."""
        from common import Watchdog
        harness.keep_dir.mkdir()
        assert harness.run('write_satellite_config').returncode == 0
        assert yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())[Watchdog.KEY] is True

    def test_two_installs_get_two_tokens(self, harness):
        harness.keep_dir.mkdir()
        tokens = set()
        for _ in range(2):
            assert harness.run('write_satellite_config').returncode == 0
            tokens.add(yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())['imessage_satellite']['token'])
        assert len(tokens) == 2

    def test_the_release_line_and_the_telemetry_are_written_as_for_a_server(self, harness):
        harness.keep_dir.mkdir()
        assert harness.run('init_telemetry; write_satellite_config').returncode == 0
        config = yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())
        assert config['developer_release'] is True                      # the harness installs from master
        assert config['telemetry']['enabled'] is True and re.fullmatch(r'[0-9a-f-]{36}', config['telemetry']['id'])
        assert harness.run('BRANCH=release; init_telemetry; write_satellite_config', KEEP_NO_TELEMETRY='1').returncode == 0
        config = yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())
        assert config['developer_release'] is False and config['telemetry'] == {'enabled': False}

    @pytest.mark.parametrize('env, answers, expected, asked', [
        ({}, None, False, 0),
        ({'KEEP_LAN_ACCESS': '1'}, ['2'], True, 0),
        ({'KEEP_LAN_ACCESS': '0'}, ['2'], False, 0),
        ({}, [''], False, 1),
        ({}, ['1'], False, 1),
        ({}, ['2'], True, 1),
        ({}, ['yes', ' 2 '], True, 2),
        ({}, ['a', 'b', 'c', '2'], False, 3),
    ], ids=['no terminal', 'the variable on', 'the variable off', 'Enter', 'this Mac', 'another Mac',
            'a wrong answer first', 'three wrong answers'])
    def test_whether_it_answers_the_network_is_asked_once_and_closed_by_default(self, harness, env, answers, expected, asked):
        harness.keep_dir.mkdir()
        if answers is not None:
            harness.set_answers(*answers)
        result = harness.run(('TTY=/dev/tty; ' if answers is not None else '') + 'write_satellite_config', **env)
        assert result.returncode == 0, result.stdout
        config = yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())
        assert config['imessage_satellite']['lan_access'] is expected
        assert len(harness.prompts) == asked

    def test_a_lan_variable_that_is_neither_yes_nor_no_stops_the_install(self, harness):
        harness.keep_dir.mkdir()
        result = harness.run('write_satellite_config', KEEP_LAN_ACCESS='maybe')
        assert result.returncode == 1
        assert not (harness.keep_dir / 'config.yaml').exists()

    def test_the_summary_carries_what_the_server_needs_to_pair_and_the_grants(self, harness):
        """Contract with the server's config keys (common/imessage/Hosts): what is printed is what
        the owner types into config_request there."""
        from common.imessage import Hosts
        result = harness.run(MAIN, KEEP_ROLE=SATELLITE, KEEP_NO_TELEMETRY='1')
        assert result.returncode == 0, result.stdout
        token = yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())['imessage_satellite']['token']
        said = result.stdout
        lines = said.splitlines()
        host_line = next(line for line in lines if f'imessage.{Hosts.HOST_KEY} ' in line)
        token_line = next(line for line in lines if f'imessage.{Hosts.TOKEN_KEY} ' in line)
        assert 'http://127.0.0.1:8898' in host_line
        assert token in token_line
        # ... and for the assistant's own Apple ID the same two values go under the chat interface's keys
        # (interfaces/imessage/Allowlist.SECTION): an owner pairing that account must not be sent to imessage.*
        from interfaces.imessage.Allowlist import SECTION
        chat_host = next(line for line in lines if f'{SECTION}.{Hosts.HOST_KEY} ' in line)
        chat_token = next(line for line in lines if f'{SECTION}.{Hosts.TOKEN_KEY} ' in line)
        assert 'http://127.0.0.1:8898' in chat_host and token in chat_token
        assert 'secret' in said
        assert 'Full Disk Access' in said and 'Automation' in said
        assert 'Accessibility' in said and 'unlocked' in said, "the typing dots' grant and what else they need"
        assert token not in (harness.keep_dir / 'log' / 'install.log').read_text(), "the token reached the install log"

    def test_a_satellite_for_a_server_on_another_mac_is_paired_by_this_macs_name(self, harness):
        from common.imessage import RemoteMessagesHost
        existing_install(harness, EXISTING_SATELLITE.replace('lan_access: false', 'lan_access: true'))
        result = harness.run('scutil() { echo keep-mini; }; ROLE=imessage_satellite; summary')
        address = re.search(r'imessage\.host\s+(\S+)', result.stdout).group(1)
        assert address == 'http://keep-mini.local:8901'
        assert re.search(r'imessage_chat\.host\s+(\S+)', result.stdout).group(1) == address
        assert RemoteMessagesHost.RemoteMessagesHost(address, 'Zq4nB8cR2yH6sD0fG3jA7vK1mX5pL9wT').address == address, \
            "the server would not take the printed address"
        assert 'http://127.0.0.1:8901' in result.stdout, "a server on this same Mac is not told its address"

    @pytest.mark.parametrize('env', [{}, {'KEEP_ROLE': SATELLITE}], ids=['the bare command', 'with the switch'])
    def test_a_rerun_on_a_satellite_keeps_its_config_and_shows_the_pairing_again(self, harness, env):
        existing_install(harness, EXISTING_SATELLITE)
        result = harness.run(MAIN, **env)
        assert result.returncode == 0, result.stdout
        assert (harness.keep_dir / 'config.yaml').read_text() == EXISTING_SATELLITE
        assert 'obtain_api_key' not in ran(result) and 'launch' in ran(result)
        assert len(steps(result)) == steps(result)[-1][1]
        assert 'http://127.0.0.1:8901' in result.stdout and 'Zq4nB8cR2yH6sD0fG3jA7vK1mX5pL9wT' in result.stdout
        assert not harness.curl_calls.exists(), "the config's telemetry opt-out was not honoured"

    @pytest.mark.parametrize('config, role', [(EXISTING_SERVER, SATELLITE), (EXISTING_SATELLITE, 'server')],
                             ids=['a satellite over a server', 'a server over a satellite'])
    def test_an_install_is_never_turned_into_the_other_role(self, harness, config, role):
        existing_install(harness, config)
        result = harness.run(MAIN, KEEP_ROLE=role)
        assert result.returncode == 1
        assert ran(result) == []
        assert (harness.keep_dir / 'config.yaml').read_text() == config
        assert 'watchdog' in result.stdout, "a second folder under the same user shares one watchdog agent"

    def test_a_satellite_over_a_server_is_sent_to_another_user_or_mac_not_to_a_second_folder(self, harness):
        existing_install(harness, EXISTING_SERVER)
        result = harness.run(MAIN, KEEP_ROLE=SATELLITE)
        said = result.stdout
        assert 'another macOS user' in said and 'another Mac' in said
        assert not re.search(r'KEEP_ROLE=\S+ KEEP_DIR=|KEEP_DIR=\S+ KEEP_ROLE=', said), \
            "the refusal still hands out a command for a second folder under this user"

    def test_a_rerun_on_a_server_is_what_it_was(self, harness):
        existing_install(harness, EXISTING_SERVER)
        result = harness.run(MAIN)
        assert result.returncode == 0, result.stdout
        assert (harness.keep_dir / 'config.yaml').read_text() == EXISTING_SERVER
        assert steps(result)[-1] == (8, 8)
        assert 'imessage.host' not in result.stdout

    MAC = ('sysctl() { echo 1; }; sw_vers() { echo 15.1; }; MIN_FREE_GB=0; MIN_MACOS=14\n'
           'uname() { if [ "$1" = -m ]; then echo arm64; else echo Darwin; fi; }\n')

    def test_the_mac_check_does_not_ask_for_anthropic_on_a_satellite(self, harness):
        result = harness.run(self.MAC + 'ROLE=imessage_satellite; check_mac')
        assert result.returncode == 0, result.stdout
        reached = harness.curl_calls.read_text()
        assert 'github.com' in reached and 'anthropic' not in reached

    def test_the_mac_check_still_asks_for_anthropic_on_a_server(self, harness):
        result = harness.run(self.MAC + 'check_mac')
        assert result.returncode == 0, result.stdout
        assert 'api.anthropic.com' in harness.curl_calls.read_text()

    def test_a_satellites_install_events_say_so_and_a_servers_are_as_they_were(self, harness):
        event = 'TELEMETRY_ID=11111111-2222-4333-8444-555555555555; telemetry_started'
        harness.run('ROLE=imessage_satellite; ' + event)
        harness.run(event)
        satellite, server = (json.loads(re.search(r'-d (\{.*\}) https://', line).group(1))['properties']
                             for line in harness.curl_calls.read_text().splitlines())
        assert satellite['role'] == SATELLITE
        assert {**satellite, 'role': None, 'default_dir': None} == {**server, 'role': None, 'default_dir': None}
        assert 'role' not in server


# main() on Linux, with the steps that touch the machine or the network replaced as in MAIN. The
# Mac's two steps are named too: on Linux neither may run.
LINUX_MAIN = MAIN.replace('for step in ', 'for step in check_linux ensure_packages find_browser ')
ARCH = os.uname().machine
CHECK = 'uname() { echo x86_64; }; MIN_FREE_GB=0; check_linux'
NO_PYTHON = 'find_python() { return 1; }\n'
A_PYTHON = 'find_python() { return 0; }\n'
EVERY_PACKAGE = ['git', 'curl', 'ca-certificates', 'ffmpeg', 'iproute2', 'procps', 'libsndfile1', 'libmagic1',
                 'openssh-client']


def on_linux(harness, script: str, **env):
    return harness.run(script, KEEP_OS=LINUX, **env)


class TestTheLinuxMachineCheck:
    """Ubuntu and Debian on x86_64 or aarch64 with a systemd user session; anything else stops
    before a package is asked for."""

    @pytest.mark.parametrize('release', [UBUNTU, DEBIAN], ids=['Ubuntu', 'Debian'])
    def test_ubuntu_and_debian_pass_without_a_warning(self, harness, release):
        harness.os_release.write_text(release)
        result = on_linux(harness, CHECK)
        assert result.returncode == 0, result.stdout
        assert 'warn:' not in harness.log.read_text()
        assert 'api.anthropic.com' in harness.calls('curl')[-1]

    def test_a_distribution_built_on_debian_installs_with_a_warning(self, harness):
        harness.os_release.write_text(MINT)
        result = on_linux(harness, CHECK)
        assert result.returncode == 0, result.stdout
        assert 'warn:' in harness.log.read_text() and 'Linux Mint 22' in result.stdout

    def test_an_older_release_than_the_ones_tried_is_warned_about(self, harness):
        harness.os_release.write_text(UBUNTU.replace('24.04', '22.04'))
        result = on_linux(harness, CHECK)
        assert result.returncode == 0, result.stdout
        assert 'warn:' in harness.log.read_text()

    @pytest.mark.parametrize('release', [FEDORA, None], ids=['Fedora', 'no os-release'])
    def test_another_distribution_stops_the_install(self, harness, release):
        if release is None:
            harness.os_release.unlink()
        else:
            harness.os_release.write_text(release)
        result = on_linux(harness, CHECK + '; ensure_packages')
        assert result.returncode == 1
        assert harness.calls('sudo') == [] and harness.calls('systemctl') == []
        assert 'Ubuntu and Debian' in result.stdout and 'Unexpected error' not in result.stdout
        assert release is None or 'Fedora' in result.stdout

    def test_os_release_is_read_and_never_run(self, harness):
        harness.os_release.write_text(UBUNTU + f'KEEP_DIR=/srv/elsewhere\ntouch "{harness.dir}/ran"\n')
        result = on_linux(harness, CHECK + '; printf "%s" "$KEEP_DIR"')
        assert result.stdout.endswith(str(harness.keep_dir))
        assert not (harness.dir / 'ran').exists()

    @pytest.mark.parametrize('arch', ['armv7l', 'i686', 'riscv64'])
    def test_an_architecture_without_wheels_stops_the_install(self, harness, arch):
        result = on_linux(harness, CHECK.replace('x86_64', arch))
        assert result.returncode == 1
        assert arch in result.stdout

    @pytest.mark.parametrize('arch', ['x86_64', 'aarch64'])
    def test_both_supported_architectures_pass(self, harness, arch):
        result = on_linux(harness, CHECK.replace('x86_64', arch))
        assert result.returncode == 0, result.stdout

    def test_no_systemd_user_session_stops_with_systemds_own_reason(self, harness):
        """A container, WSL without systemd, a shell reached through su: start.sh could never
        start the unit, so the install must not get as far as apt."""
        harness.flag('no_user_systemd')
        result = on_linux(harness, CHECK + '; ensure_packages')
        assert result.returncode == 1
        assert 'Failed to connect to bus' in result.stdout
        assert all(call.startswith('--user ') for call in harness.calls('systemctl'))
        assert harness.calls('sudo') == []

    def test_a_machine_without_systemctl_stops(self, harness):
        harness.absent('systemctl')
        result = on_linux(harness, CHECK)
        assert result.returncode == 1 and 'systemd' in result.stdout

    def test_without_curl_the_network_is_tested_once_apt_has_brought_it(self, harness):
        """wget -qO- ... | bash on a Debian without curl: "cannot reach github.com" would be a lie."""
        harness.absent('curl')
        harness.installed(*(set(EVERY_PACKAGE) - {'curl'}))
        result = on_linux(harness, A_PYTHON + CHECK + '; echo "asked=$(cat "' + str(harness.curl_calls) + '" 2>/dev/null | wc -l)"; ensure_packages')
        assert result.returncode == 0, result.stdout
        assert re.search(r'asked=\s*0', result.stdout), "the network was tested with a curl that is not there"
        assert harness.apt_installs == [['curl']]
        assert 'github.com' in harness.curl_calls.read_text()


class TestTheAptStep:
    """One `sudo apt-get install` of what is missing — shown before sudo asks — and no sudo at
    all when nothing is."""

    def test_only_what_is_missing_is_installed_in_one_command_that_was_shown(self, harness):
        harness.installed('git', 'curl', 'ca-certificates', 'procps', 'libmagic1', 'openssh-client')
        result = on_linux(harness, A_PYTHON + 'ensure_packages')
        assert result.returncode == 0, result.stdout
        assert harness.apt_installs == [['ffmpeg', 'iproute2', 'libsndfile1']]
        assert 'sudo apt-get install -y --no-install-recommends ffmpeg iproute2 libsndfile1' in result.stdout
        assert harness.calls('apt_get') == [], "apt-get ran outside the one sudo command"

    def test_nothing_missing_asks_for_no_sudo(self, harness):
        harness.installed(*EVERY_PACKAGE)
        result = on_linux(harness, A_PYTHON + 'ensure_packages')
        assert result.returncode == 0, result.stdout
        assert harness.calls('sudo') == []

    def test_a_library_installed_under_its_64_bit_time_name_is_not_asked_for_again(self, harness):
        """Ubuntu 24.04 has libmagic1 as libmagic1t64: a re-run must not ask for sudo forever."""
        harness.installed(*(set(EVERY_PACKAGE) - {'libmagic1'}), 'libmagic1t64')
        result = on_linux(harness, A_PYTHON + 'ensure_packages')
        assert result.returncode == 0, result.stdout
        assert harness.calls('sudo') == []

    @pytest.mark.parametrize('python, offered, expected', [
        (NO_PYTHON, True, ['python3.12', 'python3.12-venv']),
        (NO_PYTHON, False, []),
        (A_PYTHON, True, []),
    ], ids=['Ubuntu without a usable 3.12', 'Debian, which has none to offer', 'a 3.12 that can make a venv'])
    def test_the_distributions_python_is_added_only_where_there_is_one_and_it_is_needed(self, harness, python, offered, expected):
        harness.installed(*(set(EVERY_PACKAGE) - {'ffmpeg'}))
        if offered:
            harness.flag('apt_offers_python')
        result = on_linux(harness, python + 'ensure_packages')
        assert result.returncode == 0, result.stdout
        assert harness.apt_installs == [['ffmpeg', *expected]]

    def test_a_python_without_venv_gets_only_the_missing_half(self, harness):
        """Ubuntu 24.04 as it comes: python3.12 is there, python3.12-venv is not."""
        harness.installed(*EVERY_PACKAGE, 'python3.12')
        harness.flag('apt_offers_python')
        result = on_linux(harness, NO_PYTHON + 'ensure_packages')
        assert result.returncode == 0, result.stdout
        assert harness.apt_installs == [['python3.12-venv']]

    def test_without_sudo_the_list_is_shown_and_the_install_stops(self, harness):
        harness.absent('sudo')
        harness.installed('git', 'curl')
        result = on_linux(harness, A_PYTHON + 'ensure_packages')
        assert result.returncode == 1
        assert harness.calls('sudo') == []
        assert 'ca-certificates ffmpeg iproute2 procps libsndfile1 libmagic1 openssh-client' in result.stdout

    def test_a_sudo_that_needs_a_password_with_no_terminal_installs_nothing(self, harness):
        harness.flag('sudo_needs_password')
        harness.installed(*(set(EVERY_PACKAGE) - {'ffmpeg'}))
        result = on_linux(harness, A_PYTHON + 'ensure_packages')
        assert result.returncode == 1
        assert harness.apt_installs == []
        assert 'apt-get install -y --no-install-recommends ffmpeg' in result.stdout

    def test_a_failed_apt_stops_the_install(self, harness):
        harness.flag('apt_fails')
        harness.installed('git')
        result = on_linux(harness, A_PYTHON + 'ensure_packages; echo went-on')
        assert result.returncode == 1 and 'went-on' not in result.stdout
        assert 'Unable to locate package' in result.stdout, "apt's own words are not shown"

    def test_empty_package_lists_are_refreshed_once_and_the_install_goes_on(self, harness):
        """A new cloud image (Ubuntu's and Debian's alike) has no lists: apt knows no package."""
        harness.flag('apt_lists_empty')
        harness.installed(*(set(EVERY_PACKAGE) - {'ffmpeg'}))
        result = on_linux(harness, A_PYTHON + 'ensure_packages; echo went-on')
        assert result.returncode == 0 and 'went-on' in result.stdout, result.stdout
        assert [call for call in harness.calls('sudo') if ' update' in call] == ['env DEBIAN_FRONTEND=noninteractive apt-get update']
        assert harness.apt_installs == [['ffmpeg'], ['ffmpeg']]
        assert 'sudo apt-get update' in result.stdout, "the second command run as root was not shown"

    def test_the_refreshed_lists_may_offer_the_python_they_could_not_before(self, harness):
        harness.flag('apt_lists_empty')
        harness.flag('apt_offers_python')
        harness.installed(*(set(EVERY_PACKAGE) - {'ffmpeg'}))
        result = on_linux(harness, NO_PYTHON + 'ensure_packages')
        assert result.returncode == 0, result.stdout
        assert harness.apt_installs == [['ffmpeg'], ['ffmpeg', 'python3.12', 'python3.12-venv']]

    def test_an_apt_that_fails_for_another_reason_is_not_answered_with_an_update(self, harness):
        harness.flag('apt_broken')
        harness.installed('git')
        result = on_linux(harness, A_PYTHON + 'ensure_packages')
        assert result.returncode == 1
        assert not any(' update' in call for call in harness.calls('sudo'))
        assert 'dpkg was interrupted' in result.stdout, "apt's own words are not shown"

    def test_an_apt_that_reports_success_and_installed_nothing_stops_the_install(self, harness):
        """What dpkg has is asked again afterwards: the venv and the server are not built on
        packages apt only said it installed."""
        harness.installed(*(set(EVERY_PACKAGE) - {'ffmpeg', 'libsndfile1'}))
        result = on_linux(harness, A_PYTHON + 'sudo() { return 0; }\nensure_packages; echo went-on')
        assert result.returncode == 1 and 'went-on' not in result.stdout
        assert 'Unexpected error' not in result.stdout
        still = result.stdout.rsplit('still not installed', 1)[1]
        assert 'ffmpeg' in still and 'libsndfile1' in still and 'git' not in still

    def test_the_packages_are_the_ones_ci_and_the_plan_name(self):
        """Contract with the system: what start.sh needs (ss, pgrep), what the Python packages
        load (libsndfile, libmagic) and what the install itself runs."""
        packages = re.search(r'^APT_PACKAGES="(.*)"$', INSTALL_SH.read_text(), re.M).group(1).split()
        assert set(packages) == set(EVERY_PACKAGE)
        workflow = (INSTALL_SH.parent / '.github' / 'workflows' / 'linux.yml').read_text()
        ci = re.search(r'apt-get install .*', workflow).group(0).split()
        assert {'libmagic1', 'libsndfile1'} <= set(ci) and {'libmagic1', 'libsndfile1'} <= set(packages)


class TestPythonOnLinux:
    def test_uvs_python_is_found_under_the_name_uv_gives_this_machines_build(self, harness):
        """Contract with uv's folder names (cpython-<version>-<os>-<arch>-<libc>): the Mac's build
        beside it is not this machine's."""
        home = harness.dir / 'home'
        ours = home / '.local/share/uv/python' / f'cpython-3.12.7-linux-{ARCH}-gnu' / 'bin' / 'python3.12'
        macs = home / '.local/share/uv/python' / 'cpython-3.12.7-macos-aarch64-none' / 'bin' / 'python3.12'
        for python in (ours, macs):
            python.parent.mkdir(parents=True)
            python.write_text('')
        harness.absent('python3.12')
        accept_under_home = 'python_ok() { case "$1" in "$HOME"/*) return 0 ;; esac; return 1; }\n'
        linux = on_linux(harness, accept_under_home + 'find_python; printf "%s" "$PY"', HOME=str(home))
        assert linux.stdout.endswith(str(ours))
        mac = harness.run(accept_under_home + 'find_python; printf "%s" "$PY"', HOME=str(home))
        assert mac.stdout.endswith(str(macs))

    def test_a_python_that_cannot_make_a_venv_is_not_usable_on_linux(self, harness):
        """Debian and Ubuntu ship ensurepip apart: python_ok asks the candidate for it there, and
        for the machine's own architecture rather than the Mac's arm64."""
        asked = harness.dir / 'python_asked'
        python = harness.dir / 'python3.12'
        python.write_text(f'#!/bin/bash\necho "$*" > "{asked}"\n')
        python.chmod(0o755)
        on_linux(harness, f'python_ok "{python}"')
        assert asked.read_text().split() == ['-', ARCH, 'ensurepip']
        harness.run(f'python_ok "{python}"')
        assert asked.read_text().split() == ['-', 'arm64']

    def test_homebrew_is_not_the_way_to_a_python_on_linux(self, harness):
        script = ('find_python() { return 1; }; brew() { echo "ran:brew $*"; }; uv() { :; }; run() { echo "ran:$*"; }\n'
                  'python_ok() { return 0; }; python_version() { echo 3.12.7; }\n'
                  'ensure_python')
        result = on_linux(harness, script, HOME=str(harness.dir / 'home'))
        assert 'ran:brew' not in result.stdout and 'uv python install 3.12' in result.stdout

    def test_the_voice_requirements_come_with_pytorchs_cpu_index_on_linux_only(self, harness):
        """Contract with common/RequirementsChecker.TORCH_CPU_INDEX: an update must not swap the
        build the install chose — and on x86_64 the default build is the multi-GB CUDA one."""
        from common.RequirementsChecker import TORCH_CPU_INDEX
        harness.log.write_text('')
        script = 'PIP_EXPECTED_PACKAGES=1; run() { echo "run:$*"; }; install_requirements'
        linux = next(line for line in on_linux(harness, script).stdout.splitlines() if 'requirements-voice.txt' in line)
        assert f'--extra-index-url {TORCH_CPU_INDEX} ' in linux
        mac = next(line for line in harness.run(script).stdout.splitlines() if 'requirements-voice.txt' in line)
        assert 'index-url' not in mac


class TestWhereALinuxInstallIsRefused:
    def test_a_satellite_is_a_mac(self, harness):
        result = on_linux(harness, LINUX_MAIN, KEEP_ROLE=SATELLITE, KEEP_NO_TELEMETRY='1')
        assert result.returncode == 1
        assert ran(result) == [] and 'Mac' in result.stdout
        assert not harness.keep_dir.exists(), "a folder was made for an install that is refused"

    def test_root_is_refused_before_a_folder_is_made_in_its_home(self, harness):
        """Seen on Ubuntu: the refusal came after the log was opened, and left /root/Keep/log behind."""
        as_root = 'id() { if [ "$1" = -u ]; then echo 0; else command id "$@"; fi; }; '
        result = on_linux(harness, as_root + LINUX_MAIN, KEEP_NO_TELEMETRY='1')
        assert result.returncode == 1
        assert ran(result) == [] and not harness.keep_dir.exists()

    def test_an_existing_satellite_config_is_refused_and_left_as_it_is(self, harness):
        """A folder copied over from a Mac."""
        existing_install(harness, EXISTING_SATELLITE)
        result = on_linux(harness, LINUX_MAIN)
        assert result.returncode == 1
        assert ran(result) == []
        assert (harness.keep_dir / 'config.yaml').read_text() == EXISTING_SATELLITE

    @pytest.mark.parametrize('name', ['My Keep', 'Keep 100%', "Marcus' Keep", 'Keep\\backup'])
    def test_a_folder_start_sh_would_refuse_is_refused_before_anything_is_made(self, harness, name):
        """The names test_start_sh.py has scripts/service.sh refuse: an install there would be
        cloned, built and then never start."""
        harness.keep_dir = harness.dir / name
        result = on_linux(harness, LINUX_MAIN, KEEP_NO_TELEMETRY='1')
        assert result.returncode == 1
        assert ran(result) == [] and not harness.keep_dir.exists()
        assert 'KEEP_DIR=' in result.stdout

    def test_a_plain_name_that_is_a_symlink_to_such_a_folder_is_refused_too(self, harness):
        """start.sh writes the unit for the folder it is really in (`pwd -P`)."""
        (harness.dir / 'My Keep').mkdir()
        harness.keep_dir.symlink_to(harness.dir / 'My Keep')
        result = on_linux(harness, LINUX_MAIN, KEEP_NO_TELEMETRY='1')
        assert result.returncode == 1
        assert ran(result) == [] and list((harness.dir / 'My Keep').iterdir()) == []
        assert 'systemd' in result.stdout and 'KEEP_DIR=' in result.stdout

    @pytest.mark.parametrize('name, unsafe', [*((name, True) for name in UNIT_UNSAFE_NAMES),
                                              *((name, False) for name in UNIT_SAFE_NAMES)],
                             ids=[repr(name) for name in UNIT_UNSAFE_NAMES + UNIT_SAFE_NAMES])
    def test_the_installer_refuses_exactly_the_folders_service_sh_refuses(self, harness, name, unsafe):
        """The two scripts carry the set apart (the installer runs before a checkout exists):
        test_start_sh.py asks scripts/service.sh about the same names."""
        result = on_linux(harness, 'if unit_unsafe "$FOLDER"; then printf refused; else printf fine; fi',
                          FOLDER=f'/home/marcus/{name}')
        assert result.stdout.endswith('refused' if unsafe else 'fine')

    def test_such_a_folder_is_fine_on_a_mac_and_letters_outside_ascii_on_linux(self, harness):
        harness.keep_dir = harness.dir / 'My Keep'
        assert ran(harness.run(MAIN, KEEP_NO_TELEMETRY='1'))
        harness.keep_dir = harness.dir / 'Kęp-ö'
        assert ran(on_linux(harness, LINUX_MAIN, KEEP_NO_TELEMETRY='1', LC_ALL='C'))


class TestTheLinuxInstallEndToEnd:
    def test_the_linux_steps_run_in_order_and_the_macs_two_do_not(self, harness):
        result = on_linux(harness, LINUX_MAIN, KEEP_NO_TELEMETRY='1')
        assert result.returncode == 0, result.stdout
        assert ran(result) == ['check_linux', 'ensure_packages', 'ensure_python', 'ensure_repo', 'ensure_venv',
                               'install_requirements', 'obtain_api_key', 'find_browser', 'launch']
        counted = steps(result)
        assert counted == [(n, len(counted)) for n in range(1, len(counted) + 1)]
        assert yaml.safe_load((harness.keep_dir / 'config.yaml').read_text())['vault']['key']

    def test_lingering_is_asked_for_the_installing_user_without_sudo(self, harness):
        import pwd
        on_linux(harness, LINUX_MAIN, KEEP_NO_TELEMETRY='1')
        (call,) = harness.calls('loginctl')
        assert call.split()[-2:] == ['enable-linger', pwd.getpwuid(os.getuid()).pw_name]
        assert harness.calls('sudo') == []

    def test_a_mac_install_asks_neither_systemd_nor_apt_anything(self, harness):
        result = harness.run(MAIN, KEEP_NO_TELEMETRY='1')
        assert result.returncode == 0, result.stdout
        assert not any(harness.calls(tool) for tool in ('sudo', 'apt_get', 'systemctl', 'loginctl'))

    SUMMARY = 'BROWSER=/usr/bin/chromium; LINGER=on; summary'

    def test_the_summary_says_how_to_look_at_stop_and_reach_the_service(self, harness):
        said = on_linux(harness, self.SUMMARY).stdout
        for fact in ('systemctl --user status keep', 'journalctl --user -u keep', f'{harness.keep_dir}/log/raw-floxy.log',
                     'systemctl --user disable keep.service', f'{harness.keep_dir}/stop.sh', 'ssh -t',
                     f'{harness.keep_dir}/keep.sh'):
            assert fact in said, fact
        for mac_only in ('Terminal', 'Full Disk Access', 'microphone', 'macOS'):
            assert mac_only not in said, mac_only
        assert 'enable-linger' not in said and 'apt-get' not in said

    def test_the_macs_summary_names_no_systemd(self, harness):
        said = harness.run(self.SUMMARY).stdout
        assert 'systemctl' not in said and 'ssh' not in said and f'{harness.keep_dir}/start.sh' in said
        assert 'lan_access' not in said, "the Mac's summary changed"

    @pytest.mark.parametrize('config, answers', [
        ('api_keys:\n  anthropic: x\n', 'this machine only'),
        ('interfaces:\n  lan_access: false\n', 'this machine only'),
        ('interfaces:\n  lan_access: true\n', 'the whole network'),
    ], ids=['a fresh config', 'off', 'on'])
    def test_the_summary_says_who_reaches_the_server_as_config_yaml_has_it(self, harness, config, answers):
        """A Linux server has no screen: the phone app is how most people would reach it, and it
        cannot until interfaces.lan_access is on."""
        harness.keep_dir.mkdir(parents=True, exist_ok=True)
        (harness.keep_dir / 'config.yaml').write_text(config)
        said = on_linux(harness, self.SUMMARY).stdout
        assert 'interfaces.lan_access' in said and answers in said and 'phone app' in said

    def test_a_refused_linger_is_said_with_the_command_that_turns_it_on(self, harness):
        """Without it the server stops when the ssh session that installed it ends."""
        harness.flag('linger_refused')
        said = on_linux(harness, 'enable_linger; summary').stdout
        assert 'sudo loginctl enable-linger' in said

    @pytest.mark.parametrize('release, arch, advice, unwanted', [
        (UBUNTU, 'x86_64', 'google-chrome-stable_current_amd64.deb', 'snap'),
        (DEBIAN, 'aarch64', 'apt-get install -y chromium', 'google-chrome'),
        (UBUNTU, 'aarch64', 'snap', 'amd64'),
    ], ids=['x86_64', 'Debian on ARM', 'Ubuntu on ARM'])
    def test_without_a_browser_the_summary_says_what_to_install_and_installs_nothing(self, harness, release, arch, advice, unwanted):
        harness.os_release.write_text(release)
        harness.absent(*BROWSERS)
        result = on_linux(harness, f'uname() {{ echo {arch}; }}; find_browser; summary')
        assert result.returncode == 0, result.stdout
        assert advice in result.stdout and unwanted not in result.stdout
        assert harness.calls('sudo') == [] and harness.calls('apt_get') == []

    def test_a_browser_webaccess_would_find_is_found(self, harness):
        """Contract with WebAccess, which looks for Chrome under the same names."""
        from assistant.tools.WebAccess import WebAccess
        assert list(WebAccess._CHROME_PATH_NAMES) == BROWSERS
        harness.absent(*BROWSERS[:-1])
        result = on_linux(harness, f'{BROWSERS[-1]}() {{ :; }}; find_browser; summary')
        assert 'apt-get' not in result.stdout and BROWSERS[-1] in result.stdout

    def start_script(self, harness) -> Path:
        """A start.sh that only says it ran, and whether it was given a terminal."""
        harness.keep_dir.mkdir()
        started = harness.dir / 'started'
        script = harness.keep_dir / 'start.sh'
        script.write_text(f'#!/bin/bash\nif [ -t 0 ]; then echo terminal > "{started}"; else echo none > "{started}"; fi\n')
        script.chmod(0o755)
        return started

    def test_without_a_terminal_a_linux_install_still_starts_the_service(self, harness):
        """ssh box 'curl ... | ANTHROPIC_API_KEY=... bash': the unit needs no window, unlike the Mac's Terminal launch."""
        started = self.start_script(harness)
        result = on_linux(harness, 'launch')
        assert result.returncode == 0, result.stdout
        assert started.read_text().strip() == 'none'
        assert f'{harness.keep_dir}/keep.sh' in result.stdout

    def test_without_a_terminal_a_mac_install_does_not_start(self, harness):
        started = self.start_script(harness)
        assert harness.run('launch').returncode == 0
        assert not started.exists()

    def test_keep_no_start_starts_nothing_on_linux_either(self, harness):
        started = self.start_script(harness)
        assert on_linux(harness, 'launch', KEEP_NO_START='1').returncode == 0
        assert not started.exists()


class TestLinuxTelemetry:
    EVENT = 'TELEMETRY_ID=11111111-2222-4333-8444-555555555555; telemetry_started'

    def properties(self, harness) -> dict:
        (line,) = harness.curl_calls.read_text().splitlines()
        return json.loads(re.search(r'-d (\{.*\}) https://', line).group(1))['properties']

    def test_the_chip_is_the_architecture_and_the_cpu_model(self, harness):
        (harness.dir / 'cpuinfo').write_text('processor\t: 0\nvendor_id\t: GenuineIntel\n'
                                             'model name\t: Fictional Core 47L @ 2.40GHz\nflags\t\t: fpu vme\n')
        on_linux(harness, self.EVENT)
        properties = self.properties(harness)
        assert properties['chip'] == f'{ARCH} Fictional Core 47L @ 2.40GHz'
        assert properties['os'] == 'Linux' and properties['distribution'] == 'Ubuntu 24.04.1 LTS'
        assert 'macos' not in properties

    def test_a_board_whose_cpuinfo_names_no_model_reports_the_architecture(self, harness):
        (harness.dir / 'cpuinfo').write_text('processor\t: 0\nBogoMIPS\t: 50.00\nFeatures\t: fp asimd\n')
        on_linux(harness, self.EVENT)
        assert self.properties(harness)['chip'] == ARCH

    def test_an_install_id_is_minted_where_there_is_no_uuidgen(self, harness):
        """Debian has uuidgen in a package of its own; the id must not come out empty, or the
        install is never heard of."""
        (harness.dir / 'uuid').write_text('11111111-2222-4333-8444-555555555555\n')
        result = on_linux(harness, f'uuidgen() {{ return 127; }}; KERNEL_UUID="{harness.dir}/uuid"; init_telemetry; echo "id=$TELEMETRY_ID"')
        assert 'id=11111111-2222-4333-8444-555555555555' in result.stdout
