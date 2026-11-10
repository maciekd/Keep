"""scripts/role.sh, sourced by start.sh, stop.sh, autostart.sh and keep.sh: what the install runs
and on which ports, read from config.yaml with awk — and read the way the Python side reads the
same keys, since a script that took for a server what floxy_ai_server.py starts as a satellite
(or refuses to start at all) would stop the wrong processes.

Each case writes a config.yaml into a tmp tree, sources the script there and prints its verdict;
the Python verdict comes from a real ConfigManager on the same file.

Which processes are this checkout's is asked of the Mac's real `lsof` and `pgrep` here, about
`sleep` processes the test starts: the staged process table of the start and stop tests cannot
show how lsof spells a folder."""
import itertools
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from common import Role
from common.ConfigManager import ConfigManager
from common.imessage import Satellite

ROLE_SH = Path(__file__).resolve().parents[2] / 'scripts' / 'role.sh'
SHOW = '. scripts/role.sh; printf "%s|%s|%s|%s|%s" "$role" "$listen_ports" "$up_port" "$console_port" "$role_problem"'


def script_reading(tmp_path: Path, config: str | None, **env) -> dict:
    (tmp_path / 'scripts').mkdir(exist_ok=True)
    (tmp_path / 'scripts' / 'role.sh').write_text(ROLE_SH.read_text())
    if config is not None:
        (tmp_path / 'config.yaml').write_text(config)
    out = subprocess.run(['bash', '-c', SHOW], cwd=tmp_path, env=dict(os.environ, **env), capture_output=True, text=True,
                         timeout=10)
    assert out.returncode == 0, out.stderr
    role, listen_ports, up_port, console_port, problem = out.stdout.split('|', 4)
    return {'role': role, 'listen_ports': listen_ports.split(), 'up_port': up_port, 'console_port': console_port,
            'problem': problem}


def python_reading(tmp_path: Path, config: str) -> tuple[str, int | None]:
    """(the role Role.current() names, the port Satellite would bind — None when it refuses it)."""
    path = tmp_path / 'python_config.yaml'
    path.write_text(config)
    with patch.dict(os.environ, FLOXY_CONFIG=str(path)):
        manager = ConfigManager()
    with patch.object(Role, 'config', manager), patch.object(Satellite, 'config', manager):
        return Role.current(), Satellite._port()


SERVER_CONFIGS = {
    'no role key': 'owner_name: Marcus\n',
    'an empty role': 'role:\nowner_name: Marcus\n',
    'a null role': 'role: null\n',
    'server': 'role: server\n',
    'server in capitals, with a comment': 'role: Server   # the default\n',
    'server in quotes': 'role: "server"\n',
    'a role key inside another section': 'home:\n  role: imessage_satellite\n',
    'a commented-out satellite role': '# role: imessage_satellite\nrole: server\n',
}
SATELLITE_CONFIGS = {
    'the installer\'s shape': 'role: imessage_satellite\nimessage_satellite:\n  port: 8898\n  token: x\n  lan_access: false\n',
    'another port': 'role: imessage_satellite\nimessage_satellite:\n  port: 8901   # moved\n  token: x\n',
    'no port key': 'role: imessage_satellite\nimessage_satellite:\n  token: x\n',
    'an empty port': 'role: imessage_satellite\nimessage_satellite:\n  port:\n  token: x\n',
    'no section at all': 'role: imessage_satellite\n',
    'quoted, capitals': "role: 'Imessage_Satellite'\nimessage_satellite:\n  port: 8901\n",
    'the role after its section': 'imessage_satellite:\n  port: 8901\nrole: imessage_satellite\n',
    'a port key of another section before it': 'music:\n  port: 1400\nrole: imessage_satellite\nimessage_satellite:\n  port: 8901\n',
    'the role named twice, the satellite last': 'role: server\nrole: imessage_satellite\n',
    'the role key in quotes': '"role": imessage_satellite\n',
    'a space before the colon': 'role : imessage_satellite\n',
    'after a document marker': '---\nrole: imessage_satellite\n',
    'the section and its port key in quotes': "role: imessage_satellite\n'imessage_satellite':\n  \"port\": 8901\n",
    'a port key of a section nested in it': 'role: imessage_satellite\nimessage_satellite:\n  port: 8901\n  limits:\n    port: 1234\n',
    'the section twice, the last without a port': 'role: imessage_satellite\nimessage_satellite:\n  port: 8901\nimessage_satellite:\n  token: x\n',
    'a port inside a text that runs over lines': 'role: imessage_satellite\nimessage_satellite:\n  note: "a\n    port: 9999 b"\n  port: 8901\n',
}
REFUSED_CONFIGS = {
    'a hyphen for the underscore': 'role: imessage-satellite\n',
    'half the name': 'role: satellite\n',
    'a number': 'role: 7\n',
    'a port that is a word': 'role: imessage_satellite\nimessage_satellite:\n  port: eighty\n',
    'port 0': 'role: imessage_satellite\nimessage_satellite:\n  port: 0\n',
    'a port out of range': 'role: imessage_satellite\nimessage_satellite:\n  port: 70000\n',
    'a quoted port': 'role: imessage_satellite\nimessage_satellite:\n  port: "8898"\n',
    'null in quotes': 'role: "null"\n',
    'a list under the role key': 'role:\n- imessage_satellite\n',
}
# Valid YAML that Python starts a satellite on and the awk reader does not follow. Reading any of
# these as "no role key" would make the scripts a server's — or, for the section, watch port 8898
# while another one is bound.
UNREADABLE_CONFIGS = {
    'the whole file indented': '  role: imessage_satellite\n  imessage_satellite:\n    port: 8901\n',
    'the whole file in braces': '{role: imessage_satellite}\n',
    'the role on the line under its key': 'role:\n  imessage_satellite\n',
    'the section in braces': 'role: imessage_satellite\nimessage_satellite: {port: 8901, token: x}\n',
    'the port on the line under its key': 'role: imessage_satellite\nimessage_satellite:\n  port:\n    8901\n',
    'the port merged in from another section': 'base: &base\n  port: 8901\nrole: imessage_satellite\nimessage_satellite:\n  <<: *base\n',
}


@pytest.mark.unit
class TestTheScriptsReadTheRoleAsPythonDoes:
    @pytest.mark.parametrize('config', SERVER_CONFIGS.values(), ids=SERVER_CONFIGS.keys())
    def test_a_server(self, tmp_path, config):
        read = script_reading(tmp_path, config)
        role, _ = python_reading(tmp_path, config)
        assert (read['role'], read['problem']) == (role, '') and role == Role.SERVER
        assert read['up_port'] in read['listen_ports'] and read['console_port']

    @pytest.mark.parametrize('config', SATELLITE_CONFIGS.values(), ids=SATELLITE_CONFIGS.keys())
    def test_a_satellite_and_its_port(self, tmp_path, config):
        read = script_reading(tmp_path, config)
        role, port = python_reading(tmp_path, config)
        assert (read['role'], read['problem']) == (role, '') and role == Role.IMESSAGE_SATELLITE
        assert read['listen_ports'] == [str(port)] and read['up_port'] == str(port)
        assert read['console_port'] == '', "a satellite has no console port to wait for"

    @pytest.mark.parametrize('config', REFUSED_CONFIGS.values(), ids=REFUSED_CONFIGS.keys())
    def test_what_python_refuses_to_start_on_is_a_problem_for_the_scripts_too(self, tmp_path, config):
        read = script_reading(tmp_path, config)
        role, port = python_reading(tmp_path, config)
        assert role not in Role.ROLES or port is None, "the case is not one Python refuses"
        assert read['problem']
        assert read['listen_ports'] == [], "no port is named for a role nobody can tell"

    @pytest.mark.parametrize('config', UNREADABLE_CONFIGS.values(), ids=UNREADABLE_CONFIGS.keys())
    def test_a_spelling_the_scripts_cannot_read_is_a_problem_and_never_a_guess(self, tmp_path, config):
        read = script_reading(tmp_path, config)
        role, port = python_reading(tmp_path, config)
        assert role == Role.IMESSAGE_SATELLITE and port is not None, "the case is not one Python starts a satellite on"
        assert read['problem'], f"read as {read['role']} on port {read['up_port'] or 'none'}"
        assert read['listen_ports'] == [] and read['up_port'] == ''

    @pytest.mark.parametrize('locale', ['C', 'en_US.UTF-8'])
    def test_a_byte_order_mark_is_read_past_whatever_the_callers_locale(self, tmp_path, locale):
        """An editor's mark in front of the first key: yaml skips it, and awk compares it away
        only where there is a locale — the login applet and launchd have none."""
        config = '\ufeffrole: imessage_satellite\nimessage_satellite:\n  port: 8901\n'
        read = script_reading(tmp_path, config, LC_ALL=locale)
        role, port = python_reading(tmp_path, config)
        assert (read['role'], read['up_port'], read['problem']) == (role, str(port), '')
        assert role == Role.IMESSAGE_SATELLITE

    def test_a_role_key_of_another_section_is_no_trouble_for_a_server(self, tmp_path):
        """The refusals are for what may be the root's role: a server keeps starting with a key of
        that name, or a value on the next line, somewhere under a section."""
        config = 'home:\n  role:\n    imessage_satellite\n  port:\n    8901\nowner_name: Marcus\n'
        read = script_reading(tmp_path, config)
        assert (read['role'], read['problem']) == (python_reading(tmp_path, config)[0], '')

    def test_an_install_without_a_config_is_a_server(self, tmp_path):
        """ConfigManager writes the server's skeleton when the file is missing."""
        read = script_reading(tmp_path, None)
        assert (read['role'], read['problem']) == (Role.SERVER, '')

    def test_a_satellite_shares_no_port_with_a_server(self, tmp_path):
        server = script_reading(tmp_path, 'role: server\n')
        satellite = script_reading(tmp_path, SATELLITE_CONFIGS['no port key'])
        every_server_port = {*server['listen_ports'], server['console_port']}
        assert every_server_port.isdisjoint(satellite['listen_ports'])


_marker = itertools.count(1)


def sleeper(folder: Path) -> tuple[subprocess.Popen, str]:
    """A process running from `folder`, and the pgrep pattern that finds it and no other."""
    seconds = f'300.{os.getpid()}{next(_marker)}'
    return subprocess.Popen(['/bin/sleep', seconds], cwd=folder), f'sleep 300\\.{seconds.split(".")[1]}$'


def own_processes(checkout: Path, pattern: str, **env) -> list[int]:
    (checkout / 'scripts').mkdir(exist_ok=True)
    (checkout / 'scripts' / 'role.sh').write_text(ROLE_SH.read_text())
    out = subprocess.run(['bash', '-c', f". scripts/role.sh; own_processes '{pattern}'"], cwd=checkout, env=env,
                         capture_output=True, text=True, timeout=10)
    assert out.returncode == 0, out.stderr
    return [int(pid) for pid in out.stdout.split()]


@pytest.mark.unit
class TestWhichProcessesAreThisCheckouts:
    """The login applet and launchd's watchdog run the scripts without a locale, and lsof then
    spells a folder otherwise than the shell does. A process not recognised as this checkout's
    is a second server started beside the first, or a stop.sh that stops nothing."""

    WITHOUT_A_LOCALE = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'LC_ALL': 'C'}

    @pytest.mark.parametrize('name', ['Keep', 'Kęp-ö', 'tab\tdir', 'back\\slash', 'with space'])
    def test_a_process_of_this_checkout_is_found_whatever_the_folder_is_called(self, tmp_path, name):
        checkout, elsewhere = tmp_path / name, tmp_path / 'elsewhere'
        checkout.mkdir(), elsewhere.mkdir()
        ours, pattern = sleeper(checkout)
        theirs = subprocess.Popen(ours.args, cwd=elsewhere)     # the same command line from another folder
        try:
            assert own_processes(checkout, pattern, **self.WITHOUT_A_LOCALE) == [ours.pid]
        finally:
            for process in (ours, theirs):
                process.kill()
                process.wait(timeout=5)

    def test_a_checkout_reached_through_a_symlink_is_the_same_checkout(self, tmp_path):
        checkout, link = tmp_path / 'Keep', tmp_path / 'link'
        checkout.mkdir()
        link.symlink_to(checkout)
        ours, pattern = sleeper(checkout)
        try:
            assert own_processes(link, pattern, **self.WITHOUT_A_LOCALE) == [ours.pid]
        finally:
            ours.kill()
            ours.wait(timeout=5)
