"""The root config key `role`: with `imessage_satellite` the launcher runs the satellite and
nothing of the assistant; without it, or with `server`, it runs Keep as it always did."""
import json
import os
import subprocess
import sys
import textwrap
from unittest.mock import AsyncMock, Mock, patch

import pytest

from common import Role
from common.ConfigManager import config
from common.Paths import AISERVER_DIR, SRC_DIR

TOKEN = "kT7vQ2mX9pL4wZ8nB3cR6yH1sD5fG0jA"


def role_is(value):
    real = config.get
    return patch.object(config, "get", side_effect=lambda key, default=None:
                        (default if value is None else value) if key == "role" else real(key, default))


@pytest.fixture
def launch():
    """floxy_ai_server._launch with everything it would start replaced; the mocks by name."""
    import floxy_ai_server
    server = Mock()
    with patch.object(floxy_ai_server, "_enable_crash_traces"), \
            patch.object(floxy_ai_server, "_parse_args", return_value=(8888, False)), \
            patch.object(floxy_ai_server.RequirementsChecker, "verify_or_exit") as requirements, \
            patch.object(floxy_ai_server.Migrations, "run"), \
            patch.object(floxy_ai_server, "_verify_config_or_exit") as server_config, \
            patch.object(floxy_ai_server, "build", return_value=server) as build, \
            patch.object(floxy_ai_server, "main", new=Mock(return_value="the serving")) as main, \
            patch.object(floxy_ai_server.atexit, "register"), \
            patch.object(floxy_ai_server.asyncio, "run") as run, \
            patch("common.imessage.Satellite.config_problem", return_value="") as satellite_config, \
            patch("common.imessage.Satellite.build") as satellite_build:
        satellite_build.return_value.serve = Mock(return_value="the satellite's serving")
        yield {"launch": floxy_ai_server._launch, "requirements": requirements, "server_config": server_config,
               "build": build, "main": main, "run": run, "satellite_config": satellite_config,
               "satellite_build": satellite_build}


@pytest.mark.unit
class TestTheRoleDecidesWhatStarts:
    @pytest.mark.parametrize("value", [None, "server", "Server", " server "], ids=["no key", "server", "capital", "spaced"])
    def test_without_the_role_the_server_starts_as_it_always_did(self, launch, value):
        with role_is(value):
            launch["launch"]()
        launch["requirements"].assert_called_once()
        launch["server_config"].assert_called_once()
        launch["build"].assert_called_once_with(8888, False)
        launch["run"].assert_called_once_with("the serving")
        launch["satellite_build"].assert_not_called()

    def test_the_satellite_role_builds_no_assistant_and_serves_the_satellite(self, launch):
        with role_is("imessage_satellite"):
            launch["launch"]()
        launch["build"].assert_not_called()
        launch["server_config"].assert_not_called(), "no API key and no vault passphrase are asked of a satellite"
        launch["satellite_build"].assert_called_once_with()
        launch["run"].assert_called_once_with("the satellite's serving")

    def test_the_satellite_checks_its_requirements_first_like_the_server(self, launch):
        """An update that bumped a pin relaunches through start.sh into this check, on a satellite too."""
        order = []
        launch["requirements"].side_effect = lambda: order.append("requirements")
        launch["satellite_build"].side_effect = lambda: order.append("build") or Mock()
        with role_is("imessage_satellite"):
            launch["launch"]()
        assert order == ["requirements", "build"]

    @pytest.mark.parametrize("value", ["imessage-satellite", "satellite", "imessage_satelite", 7])
    def test_a_role_that_is_none_starts_nothing_rather_than_an_assistant_by_mistake(self, launch, value):
        with role_is(value), pytest.raises(SystemExit) as stopped:
            launch["launch"]()
        assert stopped.value.code == 1
        launch["build"].assert_not_called()
        launch["satellite_build"].assert_not_called()

    def test_a_satellite_without_a_usable_token_does_not_start_and_says_which_key(self, launch):
        import floxy_ai_server
        launch["satellite_config"].return_value = "imessage_satellite.token is not set"
        with role_is("imessage_satellite"), patch.object(floxy_ai_server, "error") as said, pytest.raises(SystemExit):
            launch["launch"]()
        launch["satellite_build"].assert_not_called()
        assert any("imessage_satellite.token" in call.args[0] for call in said.call_args_list)

    def test_the_role_key_and_its_values_are_what_config_yaml_and_the_scripts_spell(self):
        """A contract with config.yaml, install.sh and start.sh: the root key `role`, `imessage_satellite` for a satellite."""
        assert (Role.KEY, Role.SERVER, Role.IMESSAGE_SATELLITE) == ("role", "server", "imessage_satellite")


SATELLITE_BOOT = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {src!r})
    import common.logging
    common.logging.disable_file_logging = True
    # nothing here may reach this Mac's own Messages: the paths lead nowhere, as in the suite's conftest
    import common.imessage.MessagesDatabase as database, common.imessage.Contacts as contacts
    from pathlib import Path
    database.CHAT_DB_PATH = {nowhere!r} + "/chat.db"
    contacts.ADDRESS_BOOK_DIR = Path({nowhere!r}) / "AddressBook"
    import floxy_ai_server
    from common import Role
    from common.imessage import Satellite
    assert Role.current() == Role.IMESSAGE_SATELLITE and Satellite.config_problem() == ""
    satellite = Satellite.build()
    trees = sorted(name for name in sys.modules if name.split(".")[0] in ("assistant", "integrations", "interfaces"))
    # the typing dots' side of the host, as a satellite nobody asked for dots uses it
    dots = satellite.host.health().typing, satellite.host.typing("iMessage;-;+46701740605", False).showing
    satellite.host.close()
    pyobjc = sorted(name for name in ("ApplicationServices", "HIServices", "Quartz") if name in sys.modules)
    Path({answer!r}).write_text(json.dumps({{"trees": trees, "port": satellite.port, "bind": satellite.bind,
                                            "pyobjc": pyobjc, "dots": dots,
                                            "routes": sorted(rule.rule for rule in satellite.service.app.url_map.iter_rules())}}))
""")


@pytest.mark.unit
def test_a_satellite_process_imports_nothing_of_the_assistant_the_integrations_or_the_interfaces(tmp_path):
    """In a process of its own, as start.sh would run it up to the point of serving: the
    launcher's imports, the role, the satellite built from a satellite's small config.yaml."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"role: imessage_satellite\nimessage_satellite:\n  port: 8897\n  token: {TOKEN}\n")
    answer = tmp_path / "booted.json"      # not stdout: the log writer's lines land there too
    script = SATELLITE_BOOT.format(src=str(SRC_DIR), nowhere=str(tmp_path / "no-such-mac"), answer=str(answer))
    result = subprocess.run([sys.executable, "-c", script], cwd=str(AISERVER_DIR), capture_output=True, text=True,
                            timeout=60, env={**os.environ, "FLOXY_CONFIG": str(config_file)})
    assert result.returncode == 0, result.stderr[-2000:]
    booted = json.loads(answer.read_text())
    assert booted["trees"] == []
    assert booted["pyobjc"] == [], "a satellite that shows no typing dots loads no Accessibility framework"
    assert booted["dots"] == ["", False]
    assert (booted["port"], booted["bind"]) == (8897, "127.0.0.1")
    assert booted["routes"] == ["/messages/<name>", "/messages/attachment/<int:attachment_id>"]
    assert TOKEN not in result.stdout and TOKEN not in result.stderr


@pytest.mark.unit
def test_the_satellites_serving_ends_like_the_servers():
    import floxy_ai_server
    satellite = Mock(serve=AsyncMock())
    with patch("common.imessage.Satellite.config_problem", return_value=""), \
            patch("common.imessage.Satellite.build", return_value=satellite):
        floxy_ai_server._serve_satellite()
    satellite.serve.assert_awaited_once()
