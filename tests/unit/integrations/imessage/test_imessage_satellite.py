"""The iMessage integration with its Messages on another Mac: which host it builds from its
config, what it asks the owner to do when the satellite is out of reach or out of step — by
the facts each text carries — and a feed read over the real wire."""
from unittest.mock import patch

import pytest

from assistant.integration import SatelliteHints
from assistant.integration.Feed import FeedSourceAway
from assistant.integration.Integration import HealthWarning
from common import Version
from common.ConfigManager import config
from common.imessage import Hosts, Wire
from common.imessage.MessagesHost import (BAD_ANSWER, HostHealth, NO_ACCESS, NO_DATABASE, NOT_CONFIGURED, REFUSED,
                                          Recipient, SatelliteStatus, SendResult, UNREACHABLE)
from common.imessage.RemoteMessagesHost import RemoteMessagesHost
from integrations.imessage.IMessageIntegration import IMessageIntegration
from tests.unit.common.imessage.satellite_fakes import (TOKEN, closed_port, fake, follower,  # noqa: F401
                                                        running_satellite, satellite, server_identity)

from .host_fakes import FakeHost, photo, received

MODULE = 'integrations.imessage.IMessageIntegration'
ADDRESS = "http://192.0.2.7:8898"
SERVER, OTHER = "5e" * 20, "0a" * 20


@pytest.fixture(autouse=True)
def reset_singleton():
    IMessageIntegration.reset_instance()
    yield
    IMessageIntegration.reset_instance()


def configured(**section):
    real = config.get
    keys = {f"imessage.{name}": value for name, value in section.items()}
    return patch.object(config, "get", side_effect=lambda key, default=None: keys[key] if key in keys else
                        default if key.startswith("imessage.") else real(key, default))


@pytest.fixture
def integration():
    """The integration over a fake host that is, as far as it can tell, a satellite's."""
    with patch.object(Hosts, "_local", None):
        m = IMessageIntegration()
    m.host = FakeHost()
    m.host.address = ADDRESS
    m.host.health_answer = HostHealth(True, contacts=1, satellite=status())
    return m


def status(commit=SERVER, wire=Wire.WIRE_VERSION, **said) -> SatelliteStatus:
    return SatelliteStatus(wire, "v2026.11.7 (0a0a0a0a, release)", "release", commit, **said)


@pytest.fixture(autouse=True)
def this_server():
    """The server runs release at SERVER."""
    with patch.object(Version, "branch", return_value="release"), patch.object(Version, "commit", return_value=SERVER), \
            patch.object(Version, "detail", return_value="v2026.11.9 (5e5e5e5e, release)"):
        yield


def failure(integration) -> Exception:
    with pytest.raises(Exception) as raised:
        integration.check_health()
    return raised.value


@pytest.mark.unit
class TestWhichHostTheConfigNames:
    def test_without_an_address_nothing_needs_configuring(self):
        with configured(), patch.object(Hosts, "_local", None):
            integration = IMessageIntegration()
            assert integration.config_keys == [] and integration.host.address == ""

    def test_a_satellite_needs_its_token_and_is_reached_at_the_address(self):
        """`imessage.host_token` is what list_integrations then shows as set or not set."""
        with configured(host=ADDRESS, host_token=TOKEN), server_identity():
            integration = IMessageIntegration()
            assert integration.config_keys == ["imessage.host_token"]
        assert isinstance(integration.host, RemoteMessagesHost) and integration.host.address == ADDRESS

    def test_the_feeds_cursor_is_kept_as_one_of_the_account_the_config_names(self):
        """What `FeedRunner` stores beside the ROWID: one value per Messages account."""
        with configured(), patch.object(Hosts, "_local", None):
            own = IMessageIntegration().cursor_source
        IMessageIntegration.reset_instance()
        with configured(host=ADDRESS, host_token=TOKEN), server_identity():
            satellites = IMessageIntegration().cursor_source
        IMessageIntegration.reset_instance()
        with configured(host=ADDRESS + "/", host_token="another-token-entirely"), server_identity():
            respelled = IMessageIntegration().cursor_source
        assert own != satellites == respelled

    def test_the_keys_are_the_ones_the_plan_fixed(self):
        """A contract with config.yaml: `imessage.host` and `imessage.host_token`."""
        with configured(host=ADDRESS):
            assert IMessageIntegration().missing_config_keys() == ["imessage.host_token"]
        assert (Hosts.HOST_KEY, Hosts.TOKEN_KEY) == ("host", "host_token")


@pytest.mark.unit
class TestASatelliteThatCannotBeUsed:
    def test_one_that_does_not_answer_is_named_with_what_to_check_on_its_mac(self, integration):
        integration.host.health_answer = HostHealth(False, UNREACHABLE, "no connection (Connection refused)")
        said = str(failure(integration))
        assert ADDRESS in said and "Connection refused" in said
        assert "imessage.host" in said and "start.sh" in said and "lan_access" in said
        assert "Full Disk Access" not in said

    def test_one_that_refuses_the_token_asks_for_it_through_the_masked_prompt(self, integration):
        integration.host.health_answer = HostHealth(False, REFUSED, "the satellite refused the token")
        said = str(failure(integration))
        assert 'config_request("imessage.host_token")' in said and "imessage_satellite.token" in said
        assert 'restart_integration("iMessage")' in said

    def test_an_address_or_token_that_cannot_work_names_both_keys(self, integration):
        integration.host.health_answer = HostHealth(False, NOT_CONFIGURED, "no token is set")
        said = str(failure(integration))
        assert "imessage.host_token" in said and "imessage.host" in said and "no token is set" in said

    def test_an_answer_that_is_not_a_satellites_points_at_the_address_and_the_versions(self, integration):
        integration.host.health_answer = HostHealth(False, BAD_ANSWER, "HTTP 200 without a wire version")
        said = str(failure(integration))
        assert ADDRESS in said and "imessage.host" in said and "without a wire version" in said

    def test_a_satellite_too_new_for_this_server_sends_the_owner_to_the_server_not_to_the_satellite(self, integration):
        """A contract with the `restart_server` command, which pulls the server's line: the
        satellite's installer would change nothing about a server that is behind."""
        integration.host.health_answer = HostHealth(False, BAD_ANSWER, "too far apart",
                                                    satellite=SatelliteStatus(Wire.WIRE_VERSION + 2))
        said = str(failure(integration))
        assert said.startswith("restart_server()") and ADDRESS in said and "installer" not in said

    def test_a_satellite_too_old_for_this_server_is_the_one_that_updates(self, integration):
        with patch.object(Wire, "WIRE_VERSION", Wire.WIRE_VERSION + 2):
            integration.host.health_answer = HostHealth(False, BAD_ANSWER, "too far apart",
                                                        satellite=SatelliteStatus(Wire.WIRE_VERSION - 2))
            said = str(failure(integration))
        assert "restart_server" not in said and "installer on the satellite's Mac" in said and ADDRESS in said

    def test_full_disk_access_is_asked_for_on_the_satellites_mac_not_the_servers(self, integration):
        integration.host.health_answer = HostHealth(False, NO_ACCESS, "cannot read /Users/keep/Library/Messages/chat.db",
                                                    satellite=status())
        problem = failure(integration)
        said = str(problem)
        assert isinstance(problem, PermissionError)
        assert "Full Disk Access" in said and ADDRESS in said and "satellite" in said
        assert SatelliteHints.full_disk_access() not in said, "the server's own hint sends the owner to the wrong Mac"
        steps = said.split("show_full_disk_access_setting()")[0]
        assert "Terminal" in steps and "start.sh" in steps, "the way out is given before the commands that do not help"

    def test_a_satellite_mac_without_messages_says_where_to_sign_in(self, integration):
        integration.host.health_answer = HostHealth(False, NO_DATABASE, "chat.db does not exist", satellite=status())
        said = str(failure(integration))
        assert ADDRESS in said and "Full Disk Access" not in said and not isinstance(failure(integration), PermissionError)

    def test_a_feed_that_cannot_open_carries_the_same_words(self, integration):
        integration.host.health_answer = HostHealth(False, REFUSED, "the satellite refused the token")
        with patch(f"{MODULE}.error"):
            assert integration.open_feed() is False
        assert 'config_request("imessage.host_token")' in integration.open_error

    def test_a_satellite_that_is_away_when_the_feed_opens_is_waited_for_not_given_up_on(self, integration):
        """A satellite sleeps, restarts, updates itself: the runner asks again; a wrong token it does not wait out."""
        integration.host.health_answer = HostHealth(False, UNREACHABLE, "no answer to health within 30 s")
        with pytest.raises(FeedSourceAway) as away:
            integration.open_feed()
        assert ADDRESS in str(away.value) and "no answer" in str(away.value)

    def test_no_contacts_on_the_satellite_is_a_grant_on_its_mac(self, integration):
        integration.host.known["ingeborg"] = Recipient(error="contact cache is empty — could not look up 'Ingeborg'",
                                                       no_contacts=True)
        with patch(f"{MODULE}.system_note") as note:
            handle, error = integration.resolve_recipient("Ingeborg")
        assert handle is None and ADDRESS in error and SatelliteHints.full_disk_access() not in error
        assert ADDRESS in note.call_args.args[1]

    def test_a_missing_automation_grant_is_one_on_the_satellites_mac(self, integration):
        integration.host.known["alice"] = Recipient("+46701740605")
        integration.host.send_answer = SendResult("not authorized to send Apple events to Messages", unauthorized=True)
        with patch(f"{MODULE}.system_note") as note:
            assert integration.send("Alice", "hi")
        assert ADDRESS in note.call_args.args[1] and "Automation" in note.call_args.args[1]


@pytest.mark.unit
class TestASatelliteThatWorksAndIsOutOfStep:
    def test_one_on_the_servers_commit_and_wire_is_simply_healthy(self, integration):
        assert integration.check_health() is True

    def test_an_update_that_failed_is_a_warning_with_its_reason(self, integration):
        integration.host.health_answer.satellite = status(OTHER, update_failure="Git pull failed: could not resolve host")
        warning = failure(integration)
        assert isinstance(warning, HealthWarning)
        assert ADDRESS in str(warning) and "could not resolve host" in str(warning) and "v2026.11.7" in str(warning)

    def test_another_wire_version_is_a_warning_naming_both_and_who_is_behind(self, integration):
        integration.host.health_answer.satellite = status(OTHER, wire=Wire.WIRE_VERSION - 1, updating=True)
        said = str(failure(integration))
        assert isinstance(failure(integration), HealthWarning)
        assert str(Wire.WIRE_VERSION - 1) in said and str(Wire.WIRE_VERSION) in said and "the satellite is behind" in said
        integration.host.health_answer.satellite = status(OTHER, wire=Wire.WIRE_VERSION + 1)
        assert "this server is behind" in str(failure(integration))

    def test_another_version_it_is_not_leaving_is_a_warning_naming_both(self, integration):
        integration.host.health_answer.satellite = status(OTHER)
        said = str(failure(integration))
        assert isinstance(failure(integration), HealthWarning)
        assert "v2026.11.7" in said and "v2026.11.9" in said and ADDRESS in said

    def test_one_that_is_updating_right_now_is_healthy(self, integration):
        integration.host.health_answer.satellite = status(OTHER, updating=True)
        assert integration.check_health() is True

    def test_a_server_on_a_branch_nobody_follows_expects_no_satellite_to_follow(self, integration):
        """A pull-request trial: the satellite stays where it is, and that is not a problem to report."""
        integration.host.health_answer.satellite = status(OTHER)
        with patch.object(Version, "branch", return_value="trial/pr-212"):
            assert integration.check_health() is True

    def test_a_satellite_too_old_to_say_what_it_runs_is_not_called_out_of_step(self, integration):
        integration.host.health_answer.satellite = SatelliteStatus(wire=Wire.WIRE_VERSION)
        assert integration.check_health() is True

    def test_this_macs_own_messages_has_no_satellite_to_be_out_of_step_with(self, integration):
        integration.host.address = ""
        integration.host.health_answer = HostHealth(True, contacts=1)
        assert integration.check_health() is True


@pytest.mark.unit
class TestOverTheRealWire:
    """The integration, a RemoteMessagesHost, HTTP on localhost, the service, a fake host: the
    feed and a send as the server and a satellite would run them."""

    @pytest.fixture
    def wired(self, satellite, fake, follower):
        follower.answer = status()
        with configured(host=satellite.address, host_token=TOKEN):
            m = IMessageIntegration()
        with patch(f"{MODULE}.is_prompt_injection", return_value=False):
            yield m
        m.host._session.close()

    def test_the_feed_reads_the_satellites_messages_and_describes_its_photos(self, wired, fake):
        fake.rows = [received(1, "Vi ses kl 7"), received(2, "Titta", attachments=[photo()])]
        fake.files[40] = b"\xff\xd8 the photo's bytes"
        with patch(f"{MODULE}.media_analyzer") as analyzer:
            analyzer.return_value.describe_bytes.return_value = "[Photo: a red boat]"
            assert wired.open_feed() is True
            batch = wired.fetch("0", backfill=False)
        assert [("Vi ses kl 7" in e.content, "a red boat" in e.content) for e in batch.entries] == [(True, False), (False, True)]
        assert batch.cursor == "2"
        assert analyzer.return_value.describe_bytes.call_args.args[0] == fake.files[40]

    def test_a_send_goes_out_through_the_satellites_messages(self, wired, fake):
        fake.known["alice"] = Recipient("+46701740605", name="Alice", full_name="Alice Smith")
        assert wired.send("Alice", "På väg") == ""
        assert ("send", "Alice", "På väg", "Marcus") in fake.calls

    def test_a_satellite_that_went_away_fails_the_fetch_for_the_runner_and_the_health_check(self, wired, satellite, fake):
        wired.host.address = closed_port()
        with pytest.raises(FeedSourceAway) as fetch:
            wired.fetch("5", backfill=False)
        assert wired.host.address in str(fetch.value) and fake.calls == []
        with pytest.raises(FeedSourceAway):
            wired.initial_cursor(1794000000.0)
        assert wired.host.address in str(failure(wired)) and not isinstance(failure(wired), HealthWarning)

    def test_a_satellite_on_a_wire_version_too_far_away_is_waited_for_not_counted_as_failing(self, wired, satellite):
        """It is updating itself to end exactly that; five counted failures in 25 s would leave
        the feed FAILED until `restart_integration`."""
        satellite.service.wire = Wire.WIRE_VERSION + 2
        with pytest.raises(FeedSourceAway, match=f"wire version {Wire.WIRE_VERSION + 2}"):
            wired.fetch("5", backfill=False)
        with pytest.raises(FeedSourceAway, match=f"wire version {Wire.WIRE_VERSION + 2}"):
            wired.open_feed()
        satellite.service.wire = Wire.WIRE_VERSION
        assert wired.open_feed() is True and wired.fetch("5", backfill=False).cursor == "5"

    def test_an_error_of_the_satellites_own_database_is_a_failure_the_runner_counts(self, wired, fake):
        """Reached, and its chat.db would not read: not a Mac that is away."""
        fake.error = "OperationalError: database is locked"
        with pytest.raises(RuntimeError, match="database is locked"):
            wired.fetch("5", backfill=False)

    def test_a_send_later_message_reported_just_before_the_satellite_went_away_is_not_lost(self, wired, fake):
        from .host_fakes import sent
        wired._scheduled_rowids = {9}
        fake.tracked = [sent(9, "Grattis på födelsedagen")]
        real_poll, fake.poll = fake.poll, lambda *a: (_ for _ in ()).throw(ConnectionResetError("satellite restarting"))
        with pytest.raises(RuntimeError):
            wired.fetch("5", backfill=False)
        fake.poll = real_poll
        batch = wired.fetch("5", backfill=False)
        assert ["Grattis på födelsedagen" in e.content for e in batch.entries] == [True]

    def test_a_healthy_satellite_on_the_servers_version_is_a_healthy_integration(self, wired, follower):
        assert wired.check_health() is True
        follower.answer = status(OTHER, update_failure="pip could not install torch==9.9")
        assert "torch==9.9" in str(failure(wired))
