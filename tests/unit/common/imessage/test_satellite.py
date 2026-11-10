"""common/imessage/Satellite: the satellite process as config.yaml describes it — what it refuses
to start on, where it listens, what it builds. Nothing here serves or reads a Messages database."""
import asyncio
import threading
from unittest.mock import AsyncMock, Mock, patch

import pytest

import common.logging as logging_module
from common.ConfigManager import config
from common.imessage import Satellite as satellite_module
from common.imessage.MessagesHost import HostHealth, NO_ACCESS
from common.imessage.Satellite import Satellite
from common.imessage.SatelliteService import SatelliteService
from tests.unit.integrations.imessage.host_fakes import FakeHost

from .satellite_fakes import TOKEN


def configured(**section):
    real = config.get
    keys = {f"imessage_satellite.{name}": value for name, value in section.items()}
    return patch.object(config, "get", side_effect=lambda key, default=None: keys[key] if key in keys else
                        default if key.startswith("imessage_satellite.") else real(key, default))


@pytest.mark.unit
class TestConfigCheck:
    def test_a_token_and_nothing_else_is_enough(self):
        with configured(token=TOKEN):
            assert satellite_module.config_problem() == ""

    @pytest.mark.parametrize("section", [
        {}, {"token": ""}, {"token": None}, {"token": "CHANGE_ME"}, {"token": "YOUR_TOKEN_HERE"},
        {"token": "kort-hemlighet"}, {"token": "en lång hemlighet med mellanslag"},
        {"token": TOKEN, "port": "eight"}, {"token": TOKEN, "port": 0},
        {"token": TOKEN, "port": 70000}, {"token": TOKEN, "port": True},
    ], ids=["no section", "empty", "blank", "the example's placeholder", "a placeholder", "a short token",
            "a token no header can carry", "a port that is a word", "port 0", "a port out of range", "a port that is a flag"])
    def test_it_does_not_start_on_a_token_or_port_that_cannot_guard_or_serve(self, section):
        with configured(**section):
            problem = satellite_module.config_problem()
        assert problem and "imessage_satellite." in problem
        assert "hemlighet" not in problem and TOKEN not in problem, "the token is never quoted back"

    @pytest.mark.parametrize("value", ["false", "no", "off", "true", "False", 0, 1, "", ["true"]])
    def test_it_does_not_start_on_a_switch_that_is_not_a_real_true_or_false(self, value):
        """A hand-edited `lan_access: "false"` is a text, and a text is true to Python: the
        satellite would answer the whole network on a line that says no."""
        with configured(token=TOKEN, lan_access=value):
            assert "imessage_satellite.lan_access" in satellite_module.config_problem()
        with configured(token=TOKEN), watchdog_key(value):
            assert "watchdog" in satellite_module.config_problem()

    @pytest.mark.parametrize("value", [True, False, None])
    def test_a_real_switch_or_none_at_all_is_no_problem(self, value):
        with configured(token=TOKEN, lan_access=value):
            assert satellite_module.config_problem() == ""
        with configured(token=TOKEN), watchdog_key(value):
            assert satellite_module.config_problem() == ""

    def test_the_port_key_is_the_one_the_plan_and_start_sh_share(self):
        """A contract with config.yaml and start.sh, which waits for this port: `imessage_satellite.port`, 8898 unset."""
        assert (satellite_module.PORT_KEY, satellite_module.DEFAULT_PORT) == ("imessage_satellite.port", 8898)
        assert (satellite_module.TOKEN_KEY, satellite_module.LAN_ACCESS_KEY) == ("imessage_satellite.token",
                                                                                 "imessage_satellite.lan_access")


@pytest.mark.unit
class TestWhereItListens:
    @pytest.mark.parametrize("section, expected", [({}, "127.0.0.1"), ({"lan_access": False}, "127.0.0.1"),
                                                   ({"lan_access": True}, "0.0.0.0")],
                             ids=["unset", "off", "on"])
    def test_only_this_mac_unless_lan_access_is_on(self, section, expected):
        with configured(**section):
            assert satellite_module.bind_host() == expected

    @pytest.mark.parametrize("value", ["false", "no", "true", 1])
    def test_nothing_but_a_real_true_opens_it_to_the_network(self, value):
        with configured(lan_access=value):
            assert satellite_module.bind_host() == "127.0.0.1"

    def test_the_servers_own_lan_switch_does_not_open_the_satellite(self):
        real = config.get
        with patch.object(config, "get", side_effect=lambda key, default=None:
                          True if key == "interfaces.lan_access" else default if key.startswith("imessage_satellite.") else real(key, default)):
            assert satellite_module.bind_host() == "127.0.0.1"

    def test_hypercorn_binds_that_address_and_the_configured_port(self):
        serve = AsyncMock()
        host = FakeHost()
        satellite = Satellite(host, SatelliteService(host, TOKEN), "127.0.0.1", 8901)
        with patch("hypercorn.asyncio.serve", serve), patch.object(satellite, "_install_signal_handlers"), \
                patch.object(satellite_module, "stay_up"):
            asyncio.run(satellite.serve())
        app, hypercorn_config = serve.call_args.args
        assert app is satellite.service.app and hypercorn_config.bind == ["127.0.0.1:8901"]
        assert host.calls == [("health", True)], "Messages was read once before serving, so macOS asks for its grants"


@pytest.mark.unit
class TestBuild:
    @pytest.fixture
    def built(self):
        masker = logging_module._masker
        host = FakeHost()
        with configured(token=TOKEN, port=8902, lan_access=True), patch.object(satellite_module.Hosts, "local", return_value=host), \
                patch.object(satellite_module, "Updater"):
            yield satellite_module.build(), host
        logging_module.set_log_masker(masker)

    def test_it_serves_this_macs_own_messages_on_the_configured_port(self, built):
        satellite, host = built
        assert satellite.host is host and satellite.service.host is host
        assert (satellite.bind, satellite.port) == ("0.0.0.0", 8902)

    def test_the_service_takes_the_configured_token_and_no_other(self, built):
        async def ask(token):
            headers = {"Authorization": f"Bearer {token}", "X-Keep-Wire": "1"}
            return (await built[0].service.app.test_client().post("/messages/scheduled", json={}, headers=headers)).status_code
        assert asyncio.run(ask(TOKEN)) == 200 and asyncio.run(ask(TOKEN.lower())) == 401

    def test_a_log_line_that_would_carry_the_token_is_masked(self, built):
        line = logging_module.masked_for_log(f"GET /?token={TOKEN} from 192.0.2.9")
        assert TOKEN not in line and "192.0.2.9" in line


@pytest.mark.unit
class TestTheHostCheckAtBoot:
    def test_a_mac_that_keeps_the_process_out_is_told_what_to_grant_there(self):
        host = FakeHost()
        host.health_answer = HostHealth(False, NO_ACCESS, "cannot read /Users/keep/Library/Messages/chat.db")
        with patch.object(satellite_module, "error") as error:
            Satellite(host, Mock(), "127.0.0.1", 8898).check_host()
        said = error.call_args.args[0]
        assert "Full Disk Access" in said and "chat.db" in said


@pytest.fixture
def mac(tmp_path):
    """The Mac as `stay_up` changes it: the launchd agent (`Watchdog.sync`) and the login item
    (`Autostart.enable`, succeeding unless a test says otherwise), and a marker of its own."""
    with patch.object(satellite_module.Watchdog, "sync", return_value=None) as watchdog, \
            patch.object(satellite_module.Autostart, "enable", return_value=None) as login_item, \
            patch.object(satellite_module, "LOGIN_ITEM_MARKER", tmp_path / "cache" / "satellite_login_item"):
        yield Mock(watchdog=watchdog, login_item=login_item)


def watchdog_key(value):
    real = config.get
    return patch.object(config, "get", side_effect=lambda key, default=None:
                        value if key == satellite_module.Watchdog.KEY else real(key, default))


@pytest.mark.unit
class TestStayingUp:
    """No assistant runs on a satellite, so the process itself does what the server's setup and
    `config_set` do: the watchdog follows its key at each start, the login item is made once."""

    @pytest.mark.parametrize("value, wanted", [(True, True), (False, False), (None, False), ("false", False), ("true", False)],
                             ids=["on", "off", "no key", "a quoted false", "a quoted true"])
    def test_the_launchd_agent_follows_the_watchdog_key_at_every_start(self, mac, value, wanted):
        with watchdog_key(value):
            satellite_module.stay_up(readable=False)
            satellite_module.stay_up(readable=True)
        assert [call.args for call in mac.watchdog.call_args_list] == [(wanted,), (wanted,)]

    def test_no_login_item_while_messages_cannot_be_read(self, mac):
        """As on the server, where the setup waits for Full Disk Access: one dialog at a time."""
        satellite_module.stay_up(readable=False)
        mac.login_item.assert_not_called()

    def test_the_login_item_is_made_once_so_one_the_user_removed_stays_removed(self, mac):
        for _ in range(3):
            satellite_module.stay_up(readable=True)
        assert mac.login_item.call_count == 1

    def test_a_login_item_that_could_not_be_made_is_tried_again_at_the_next_start(self, mac):
        mac.login_item.return_value = "the user declined to let Terminal control System Events"
        satellite_module.stay_up(readable=True)
        mac.login_item.return_value = None
        satellite_module.stay_up(readable=True)
        satellite_module.stay_up(readable=True)
        assert mac.login_item.call_count == 2

    def test_a_failure_there_never_ends_the_satellite(self, mac):
        mac.watchdog.side_effect = RuntimeError("launchd is not answering")
        satellite_module.stay_up(readable=True)      # logged, not raised: the satellite keeps serving

    def test_serving_does_not_wait_for_it(self):
        """A login item can wait a minute and a half on a dialog at the Mac's screen."""
        release, entered = threading.Event(), threading.Event()
        seen = {}

        def blocked(readable):
            seen["readable"] = readable
            entered.set()
            release.wait(5)
        host = FakeHost()
        satellite = Satellite(host, SatelliteService(host, TOKEN), "127.0.0.1", 8901)
        with patch("hypercorn.asyncio.serve", AsyncMock()), patch.object(satellite, "_install_signal_handlers"), \
                patch.object(satellite_module, "stay_up", blocked):
            asyncio.run(satellite.serve())
            served_while_blocked = entered.wait(5) and not release.is_set()
        release.set()
        assert served_while_blocked
        assert seen == {"readable": True}, "the fake host's Messages is readable, and that is what the login item waits for"

    def test_the_typing_dots_do_not_outlive_the_serving(self):
        """SIGTERM from start.sh or stop.sh ends the serving: the windows the dots opened close with it."""
        host = FakeHost()
        host.close = Mock()
        satellite = Satellite(host, SatelliteService(host, TOKEN), "127.0.0.1", 8901)
        with patch("hypercorn.asyncio.serve", AsyncMock(side_effect=lambda *a, **k: host.close.assert_not_called())), \
                patch.object(satellite, "_install_signal_handlers"), patch.object(satellite_module, "stay_up"):
            asyncio.run(satellite.serve())
        host.close.assert_called_once_with()

    def test_it_is_told_when_messages_is_not_readable(self):
        host = FakeHost()
        host.health_answer = HostHealth(False, NO_ACCESS, "cannot read chat.db")
        assert Satellite(host, Mock(), "127.0.0.1", 8898).check_host() is False
