"""IndigoIntegration: the HomeAutomation backend (the REST commands, the prompt.txt snapshot
and error log — plans/HOME_AUTOMATION_PLAN.md), the health check, and the event log as a feed
without a cursor (plans/INDIGO_FEED_PLAN.md)."""
import re
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from requests.auth import HTTPDigestAuth

from assistant.integration.Feed import FeedEntry
from assistant.integration.HomeAutomation import HomeAutomation, prune, render
from integrations.indigo.IndigoIntegration import IndigoIntegration, EVENT_TIME_NUDGE, LOG_REFINER, PROMPT_FILE, ERROR_LOG_SECTION

MODULE = "integrations.indigo.IndigoIntegration"
DATA = Path(__file__).parents[2] / "data" / "home"
NOW = datetime(2025, 3, 15, 14, 0, 0)
SECTION_FILTERS = {
    "sensors": ["Humidity", "Luminance", "Illuminance"],
    "variables": ["Humidity[", "Last", "Light[", "Motion[", "Temperature[", "Raining"],
}
CONFIG = {"indigo.host": "indigo.local", "indigo.login": "alex", "indigo.password": "hunter2",
          "indigo.section_filters": SECTION_FILTERS}


@pytest.fixture
def integration():
    IndigoIntegration.reset_instance()
    with patch(f"{MODULE}.config") as cfg:
        cfg.get.side_effect = lambda key, default=None: CONFIG.get(key, default)
        yield IndigoIntegration()
    IndigoIntegration.reset_instance()


@pytest.mark.unit
class TestContract:
    def test_reset_stops_the_feed(self, integration):
        """restart_integration Indigo must not leave the old feed thread polling over SSH."""
        with patch("assistant.integration.FeedRunner.Feeds.get_instance") as feeds:
            integration._on_reset()
        feeds.return_value.stop.assert_called_once_with("Indigo")

    def test_rest_base_url_and_auth_come_from_config_and_are_cached(self, integration):
        assert integration.base_url == "http://indigo.local:8176"
        auth = integration.auth
        assert isinstance(auth, HTTPDigestAuth)
        assert (auth.username, auth.password) == ("alex", "hunter2")
        assert integration.base_url is integration.base_url
        assert integration.auth is auth


@pytest.mark.unit
class TestCheckHealth:
    def test_probes_the_variables_endpoint_with_digest_auth(self, integration):
        with patch(f"{MODULE}.requests.get") as get:
            assert integration.check_health() is True
        get.assert_called_once_with("http://indigo.local:8176/variables.json", auth=integration.auth, timeout=10)
        get.return_value.raise_for_status.assert_called_once()

    def test_an_http_error_propagates_to_the_monitor(self, integration):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
        with patch(f"{MODULE}.requests.get", return_value=response), \
             pytest.raises(requests.HTTPError):
            integration.check_health()

@pytest.mark.unit
class TestParseEvents:
    def test_a_message_line_becomes_a_system_entry_at_todays_clock_time(self):
        entries = IndigoIntegration.parse_events("10:30:00 (3h ago) message: Kitchen light on\n", NOW)
        assert entries == [FeedEntry(datetime(2025, 3, 15, 10, 30, 0).timestamp() + EVENT_TIME_NUDGE,
                                     "Kitchen light on")]
        assert entries[0].role == "system"
        assert EVENT_TIME_NUDGE == 2

    def test_non_message_lines_are_skipped(self):
        output = ("10:30:00 (3h ago) trigger: Some trigger\n"
                  "10:31:00 (3h ago) light_on: Lamp\n"
                  "10:32:00 (3h ago) message: Kept\n")
        assert [e.content for e in IndigoIntegration.parse_events(output, NOW)] == ["Kept"]

    def test_a_clock_time_ahead_of_now_is_yesterday(self):
        entries = IndigoIntegration.parse_events("15:00:00 (future) message: Future event\n", NOW)
        assert len(entries) == 1
        assert datetime.fromtimestamp(entries[0].timestamp).day == 14

    def test_malformed_and_blank_lines_are_skipped(self):
        output = "\nmessage: no clock time\n10:30 (x) message: short clock\n\n10:30:00 (x) message: ok\n"
        assert [e.content for e in IndigoIntegration.parse_events(output, NOW)] == ["ok"]

@pytest.mark.unit
class TestFetch:
    def test_runs_the_refiner_over_ssh_and_returns_the_days_events(self, integration):
        result = Mock(returncode=0, stdout="10:30:00 (3h ago) message: Kitchen light on\n", stderr="")
        with patch(f"{MODULE}.subprocess.run", return_value=result) as run:
            batch = integration.fetch(None, backfill=True)
        cmd = run.call_args.args[0]
        assert cmd[0] == "ssh"
        assert "BatchMode=yes" in cmd and "ConnectTimeout=10" in cmd
        assert "media@indigo.local" in cmd
        assert cmd[-1] == LOG_REFINER
        assert run.call_args.kwargs["timeout"] == 15
        assert [e.content for e in batch.entries] == ["Kitchen light on"]
        assert batch.cursor is None
        assert batch.exhausted is True

    def test_a_non_zero_ssh_exit_is_an_empty_batch_not_a_failure(self, integration):
        result = Mock(returncode=255, stdout="", stderr="ssh: connect to host indigo.local: Connection refused")
        with patch(f"{MODULE}.subprocess.run", return_value=result):
            batch = integration.fetch(None, backfill=False)
        assert batch.entries == [] and batch.cursor is None and batch.exhausted is True

    def test_a_raising_ssh_call_is_an_empty_batch_not_a_failure(self, integration):
        with patch(f"{MODULE}.subprocess.run", side_effect=subprocess.TimeoutExpired("ssh", 15)):
            batch = integration.fetch(None, backfill=False)
        assert batch.entries == [] and batch.cursor is None and batch.exhausted is True


@pytest.mark.unit
class TestCommands:
    def test_turn_on_puts_is_on_to_the_encoded_device(self, integration):
        response = Mock(ok=True, text="<html>ok</html>")
        with patch(f"{MODULE}.requests.put", return_value=response) as put:
            assert integration.turn_on("Desk lamp (Office)") == ""
        put.assert_called_once_with("http://indigo.local:8176/devices/Desk%20lamp%20%28Office%29",
                                    data={"isOn": "1"}, auth=integration.auth, timeout=10)

    def test_turn_off_and_dim(self, integration):
        response = Mock(ok=True, text="")
        with patch(f"{MODULE}.requests.put", return_value=response) as put:
            assert integration.turn_off("Lamp") == ""
            assert integration.dim("Lamp", 50) == ""
        assert [c.kwargs["data"] for c in put.call_args_list] == [{"isOn": "0"}, {"brightness": "50"}]

    def test_execute_gets_the_action_with_the_execute_method(self, integration):
        response = Mock(ok=True, text="")
        with patch(f"{MODULE}.requests.get", return_value=response) as get:
            assert integration.execute("Hall & Stairs") == ""
        get.assert_called_once_with("http://indigo.local:8176/actions/Hall%20%26%20Stairs?_method=execute",
                                    auth=integration.auth, timeout=10)

    def test_timeouts_and_connection_errors_are_words_not_exceptions(self, integration):
        with patch(f"{MODULE}.requests.put", side_effect=requests.Timeout):
            assert integration.turn_on("Lamp") == "Request timed out"
        with patch(f"{MODULE}.requests.get", side_effect=requests.ConnectionError):
            assert integration.execute("Go") == "Could not connect to home automation server"
        with patch(f"{MODULE}.requests.put", side_effect=ValueError("odd")):
            assert integration.dim("Lamp", 1) == "odd"

    def test_the_response_is_indigos_error_or_nothing(self):
        parse = IndigoIntegration._parse_response
        assert parse(Mock(ok=False, text=" error ", status_code=400)) == "error"
        assert parse(Mock(ok=False, text="", status_code=404)) == "HTTP 404"
        assert parse(Mock(ok=True, text="<html><center>err</center></html>")) == "err"
        assert parse(Mock(ok=True, text="<html>all good</html>")) == ""
        assert parse(Mock(ok=True, text="<center></center>")) == ""


@pytest.mark.unit
class TestSnapshot:
    SAMPLE = (DATA / "indigo_prompt.txt").read_text()

    def _ssh(self, stdout, returncode=0):
        return patch(f"{MODULE}.subprocess.run", return_value=Mock(returncode=returncode, stdout=stdout, stderr=""))

    def test_reads_prompt_txt_over_ssh_into_the_four_lists(self, integration):
        with self._ssh(self.SAMPLE) as run:
            snap = integration.snapshot()
        cmd = run.call_args.args[0]
        assert cmd[0] == "ssh" and "media@indigo.local" in cmd and cmd[-1] == f"cat {PROMPT_FILE}"
        assert len(snap.lights) == 22 and len(snap.sensors) == 13
        assert len(snap.variables) == 44 and len(snap.actions) == 9
        assert snap.lights[0].name == "Belysning carport och entré (Ute)" and snap.lights[0].area is None
        assert snap.flag("Night") is False
        assert snap.taken_at > 0

    def test_rendered_as_today_the_golden_output(self, integration):
        """The rendered section equals what IndigoDataSectionGenerator produced from the same
        file (blank-line runs between blocks aside: the old code copied the file's)."""
        with self._ssh(self.SAMPLE):
            snap = integration.snapshot()
        rendered = render(prune(snap, [], integration.section_filters))
        golden = (DATA / "indigo_rendered.md").read_text()
        squash = lambda t: re.sub(r"\n{3,}", "\n\n", t).rstrip()
        assert squash(rendered) == squash(golden)


    def test_an_unreadable_prompt_file_raises_for_the_error_log_too(self, integration):
        with self._ssh("", returncode=255):
            with pytest.raises(RuntimeError, match="could not read"):
                integration.error_logs()

    def test_parse_sections_and_entities(self):
        sections = IndigoIntegration.parse_sections("## LIGHTS\n\"Tak (Kök)\": on\n\n## ACTIONS\n\"Go\":\nnot a line\n\n")
        assert sections == {"LIGHTS": ['"Tak (Kök)": on'], "ACTIONS": ['"Go":', "not a line"]}
        entities = IndigoIntegration.parse_entities(sections["ACTIONS"])
        assert [(e.name, e.value) for e in entities] == [("Go", "")]

    def test_an_unknown_section_is_dropped(self, integration):
        with self._ssh("## LIGHTS\n\"Lamp\": on\n## WEIRD\n\"x\": 1\n"):
            snap = integration.snapshot()
        assert [e.name for e in snap.lights] == ["Lamp"] and snap.is_empty() is False
        assert snap.variables == []

    def test_a_failed_or_empty_read_raises(self, integration):
        with self._ssh("", returncode=255), pytest.raises(RuntimeError):
            integration.snapshot()
        with self._ssh(""), pytest.raises(RuntimeError):
            integration.snapshot()


@pytest.mark.unit
class TestErrorLogs:
    def test_returns_the_error_log_section_in_display_form(self, integration):
        text = "## LIGHTS\n\"Lamp\": on\n\n## LATEST AUTOMATION ERROR LOGS\n2026-09-15 Error send \"Dörr (WC)\" failed\n\n## VARIABLES\n"
        with patch(f"{MODULE}.subprocess.run", return_value=Mock(returncode=0, stdout=text, stderr="")):
            logs = integration.error_logs()
        assert logs == f"## {ERROR_LOG_SECTION}\n2026-09-15 Error send \"Dörr [WC]\" failed"

    def test_no_section_says_so(self, integration):
        with patch(f"{MODULE}.subprocess.run", return_value=Mock(returncode=0, stdout="## LIGHTS\n", stderr="")):
            assert integration.error_logs() == "Indigo reported no error log."


@pytest.mark.unit
class TestSshConfigKeys:
    """`indigo.ssh_user`, `indigo.prompt_file` and `indigo.log_refiner` replace the built-in
    account, state file and script, so an Indigo Mac laid out differently can be read too."""
    OVERRIDES = {**CONFIG, "indigo.ssh_user": "frances", "indigo.prompt_file": "keep/state.txt",
                 "indigo.log_refiner": "python3 keep/events.py"}

    def test_the_configured_user_file_and_script_reach_both_ssh_commands(self, integration):
        with patch(f"{MODULE}.config") as cfg:
            cfg.get.side_effect = lambda key, default=None: self.OVERRIDES.get(key, default)
            with patch(f"{MODULE}.subprocess.run", return_value=Mock(returncode=0, stdout="", stderr="")) as run:
                integration.fetch(None, backfill=True)
            refiner_cmd = run.call_args.args[0]
            with patch(f"{MODULE}.subprocess.run",
                       return_value=Mock(returncode=0, stdout=TestSnapshot.SAMPLE, stderr="")) as run:
                integration.snapshot()
            prompt_cmd = run.call_args.args[0]
        assert "frances@indigo.local" in refiner_cmd and refiner_cmd[-1] == "python3 keep/events.py"
        assert "frances@indigo.local" in prompt_cmd and prompt_cmd[-1] == "cat keep/state.txt"
        assert "media@indigo.local" not in refiner_cmd + prompt_cmd
