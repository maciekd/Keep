"""The beta telemetry: what leaves the machine, under which switch, and the week after the
first start in which the turns carry their text."""
import time
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import assistant.Telemetry as telemetry_module
import common.ConfigManager as config_module
from assistant.Telemetry import Telemetry
from common import ConfigStore
from tests.unit.waiting import wait_until


@pytest.fixture
def settings(monkeypatch):
    """A config with telemetry fully on and no id yet; the project key is built in; config
    writes are recorded, not performed."""
    values = {"telemetry.enabled": True, "telemetry.id": "", "owner_name": "Marcus"}
    monkeypatch.setattr(config_module.config, "get", lambda key, default=None: values.get(key, default))
    monkeypatch.setattr(config_module.config, "enabled_integrations", lambda: ["Tesla", "Weather"])
    monkeypatch.setattr(telemetry_module, "POSTHOG_PROJECT_KEY", "phc_fictional")
    monkeypatch.setattr(ConfigStore, "set_value", Mock(return_value="saved"))
    return values


@pytest.fixture
def sent():
    return []


@pytest.fixture
def telemetry(settings, sent):
    """Started with a fake poster; `stop()` flushes, so a test stops it before asserting."""
    t = Telemetry(post=sent.append, install_age_days=lambda: 2.0)
    t.start()
    yield t
    t.stop()


def _events(sent) -> list[dict]:
    return [event for payload in sent for event in payload["batch"]]


def _by_name(sent, name) -> list[dict]:
    return [e for e in _events(sent) if e["event"] == name]


class TestSwitches:
    def test_disabled_in_config_sends_nothing(self, settings, sent):
        settings["telemetry.enabled"] = False
        t = Telemetry(post=sent.append, install_age_days=lambda: 2.0)
        t.start()
        t.user_turn("hello", "console")
        t.stop()
        assert not t.active and sent == []

    def test_developer_mode_sends_nothing(self, settings, sent, monkeypatch):
        monkeypatch.setattr("common.DevelopmentConfig.LOCAL_DEVELOPMENT_MODE", True)
        t = Telemetry(post=sent.append, install_age_days=lambda: 2.0)
        t.start()
        t.user_turn("hello", "console")
        t.heartbeat()
        t.stop()
        assert not t.active and sent == []

    def test_an_opt_out_while_running_holds_from_the_next_event_without_a_restart(self, settings, telemetry, sent):
        telemetry.user_turn("before", "console")
        assert wait_until(lambda: _by_name(sent, "user_turn"))   # posted while it was on
        settings["telemetry.enabled"] = False
        telemetry.user_turn("after", "console")
        telemetry.assistant_turn("reply", "console")
        telemetry.tab_change("Logs", "Home", 1.0)
        telemetry.integration_switch("Weather", True, "ok")
        telemetry.heartbeat()
        telemetry.stop()
        assert [e["properties"].get("query") for e in _events(sent) if e["event"] != "heartbeat"] == ["before"]
        assert len(_by_name(sent, "heartbeat")) == 1   # the one from start

    def test_events_still_queued_at_the_opt_out_are_discarded(self, settings, sent):
        t = Telemetry(post=sent.append, install_age_days=lambda: 2.0)
        t._send([{"event": "user_turn"}])
        settings["telemetry.enabled"] = False
        t._send([{"event": "assistant_turn"}])
        assert [e["event"] for e in _events(sent)] == ["user_turn"]

    def test_without_a_built_in_key_nothing_is_sent(self, settings, sent, monkeypatch):
        monkeypatch.setattr(telemetry_module, "POSTHOG_PROJECT_KEY", "")
        t = Telemetry(post=sent.append, install_age_days=lambda: 2.0)
        t.start()
        t.user_turn("hello", "console")
        t.stop()
        assert not t.active and sent == []

    def test_the_real_poster_never_runs_under_test(self, settings):
        t = Telemetry()
        t.start()
        assert not t.active


class TestInstallId:
    def test_first_start_writes_a_uuid_to_config_and_every_event_carries_it(self, telemetry, sent):
        telemetry.user_turn("hello", "console")
        telemetry.stop()
        key, written = ConfigStore.set_value.call_args.args
        assert key == "telemetry.id"
        assert uuid.UUID(written)
        assert {e["distinct_id"] for e in _events(sent)} == {written}

    def test_an_existing_id_is_kept(self, settings, sent):
        settings["telemetry.id"] = "8f1c9b2a-0000-4000-8000-000000000001"
        t = Telemetry(post=sent.append, install_age_days=lambda: 2.0)
        t.start()
        t.stop()
        ConfigStore.set_value.assert_not_called()
        assert _events(sent)[0]["distinct_id"] == settings["telemetry.id"]


class TestEvents:
    def test_start_sends_a_heartbeat_with_the_enabled_integrations(self, telemetry, sent):
        telemetry.stop()
        [heartbeat] = _by_name(sent, "heartbeat")
        assert heartbeat["properties"]["startup"] is True
        assert heartbeat["properties"]["integrations"] == ["Tesla", "Weather"]
        assert heartbeat["properties"]["content"] is True

    def test_turns_carry_the_text_while_content_is_allowed(self, telemetry, sent):
        telemetry.user_turn("turn on the lights", "VoiceToVoice")
        telemetry.assistant_turn("Done.", "VoiceToVoice", commands=["light_on"])
        telemetry.stop()
        [user] = _by_name(sent, "user_turn")
        [reply] = _by_name(sent, "assistant_turn")
        assert user["properties"]["query"] == "turn on the lights"
        assert user["properties"]["source"] == "VoiceToVoice"
        assert reply["properties"]["reply"] == "Done."
        assert reply["properties"]["commands"] == ["light_on"]
        assert reply["properties"]["proactive"] is False

    def test_vault_values_leave_as_placeholders_in_both_directions(self, telemetry, sent, monkeypatch):
        monkeypatch.setattr(telemetry_module, "redact_for_storage",
                            lambda text: text.replace("4111 1111 1111 1111", "{{secret:marcus-visa}}"))
        telemetry.user_turn("pay with 4111 1111 1111 1111", "Telegram")
        telemetry.assistant_turn("Paid with 4111 1111 1111 1111.", "Telegram")
        telemetry.assistant_turn("Card 4111 1111 1111 1111 expires soon.", "notifier", proactive=True)
        telemetry.stop()
        texts = [e["properties"].get("query") or e["properties"].get("reply")
                 for e in _events(sent) if e["event"].endswith("_turn")]
        assert len(texts) == 3
        assert all("4111" not in text and "{{secret:marcus-visa}}" in text for text in texts)

    def test_past_the_first_week_the_text_stays_home_but_the_command_names_go(self, settings, sent):
        t = Telemetry(post=sent.append, install_age_days=lambda: 7.0)
        t.start()
        t.user_turn("read my mail", "Telegram")
        t.assistant_turn("Three new messages.", "Telegram", commands=["check_mail"])
        t.stop()
        [user] = _by_name(sent, "user_turn")
        [reply] = _by_name(sent, "assistant_turn")
        assert "query" not in user["properties"]
        assert "reply" not in reply["properties"]
        assert reply["properties"]["commands"] == ["check_mail"]
        ConfigStore.set_value.assert_called_once()   # the id only, never the content key
        assert ConfigStore.set_value.call_args.args[0] == "telemetry.id"

    def test_events_go_out_in_batches_no_larger_than_the_limit_and_none_is_lost(self, settings, sent):
        t = Telemetry(post=sent.append, install_age_days=lambda: 2.0)
        t.start()
        count = telemetry_module.BATCH_LIMIT * 2 + 7
        for i in range(count):
            t.user_turn(f"turn {i}", "console")
        t.stop()
        assert all(len(payload["batch"]) <= telemetry_module.BATCH_LIMIT for payload in sent)
        assert len(sent) >= 3
        assert [e["properties"]["query"] for e in _by_name(sent, "user_turn")] == [f"turn {i}" for i in range(count)]

    def test_the_heartbeat_repeats_on_its_interval_without_the_startup_flag(self, settings, sent, monkeypatch):
        """The daily heartbeat is what tells an uninstall from a quiet day; it has to keep
        coming after the one sent at start, and say it is not a start."""
        monkeypatch.setattr(telemetry_module, "HEARTBEAT_INTERVAL", 0.002)
        t = Telemetry(post=sent.append, install_age_days=lambda: 2.0)
        t.start()
        assert wait_until(lambda: len(_by_name(sent, "heartbeat")) >= 3)
        t.stop()
        beats = _by_name(sent, "heartbeat")
        assert len(beats) >= 3
        assert [b["properties"]["startup"] for b in beats] == [True] + [False] * (len(beats) - 1)
        assert all(b["properties"]["integrations"] == ["Tesla", "Weather"] for b in beats)

    def test_a_tab_change_names_both_tabs_and_the_time_on_the_one_left(self, telemetry, sent):
        telemetry.tab_change("Integrations/Arlo", "Home", 12.34)
        telemetry.stop()
        (event,) = _by_name(sent, "tab_change")
        assert (event["properties"]["tab"], event["properties"]["from"]) == ("Integrations/Arlo", "Home")
        assert event["properties"]["dwell"] == pytest.approx(12.3)

    def test_an_integration_switch_is_one_event_per_direction_with_the_state_it_landed_in(self, telemetry, sent):
        telemetry.integration_switch("Ring", True, "not_configured")
        telemetry.integration_switch("Ring", False, "disabled")
        telemetry.stop()
        (on,), (off,) = _by_name(sent, "integration_enabled"), _by_name(sent, "integration_disabled")
        assert (on["properties"]["integration"], on["properties"]["state"]) == ("Ring", "not_configured")
        assert (off["properties"]["integration"], off["properties"]["state"]) == ("Ring", "disabled")

    def test_a_failed_batch_is_dropped_and_the_next_one_goes_through(self, settings):
        posts = []

        def post(payload):
            posts.append(payload)
            if len(posts) == 1:
                raise ConnectionError("down")

        t = Telemetry(post=post, install_age_days=lambda: 2.0)
        t.start()
        t.stop()          # the startup heartbeat fails
        t2 = Telemetry(post=post, install_age_days=lambda: 2.0)
        t2.start()
        t2.user_turn("hello", "console")
        t2.stop()
        assert len(posts) >= 2
        assert len(_by_name(posts, "user_turn")) == 1


class TestQueueLimit:
    def test_events_past_the_queue_limit_are_dropped_and_counted_without_an_error_reaching_the_caller(
            self, settings, sent, monkeypatch):
        monkeypatch.setattr(telemetry_module, "QUEUE_LIMIT", 2)
        monkeypatch.setattr(Telemetry, "active", property(lambda self: True))   # on, but no worker drains the queue
        t = Telemetry(post=sent.append, install_age_days=lambda: 2.0)
        for n in range(5):
            t.user_turn(f"turn {n}", "console")
        assert t._dropped == 3 and t._queue.qsize() == 2
        assert sent == []


class TestStatus:
    def test_a_status_carries_every_running_feed_as_it_stands_and_the_integrations_on(self, telemetry, sent):
        telemetry.feed_count("iCloud Photos", 10)
        telemetry.feed_count("iCloud Photos", 25)
        telemetry.feed_count("iMessage", 400)
        telemetry.feed_count("Telegram", 3)
        telemetry.feed_stopped("Telegram")
        telemetry.status()
        telemetry.stop()
        [status] = _by_name(sent, "status")
        assert status["properties"]["feeds"] == {"iCloud Photos": 25, "iMessage": 400}
        assert status["properties"]["integrations"] == ["Tesla", "Weather"]

    def test_an_unchanged_status_is_not_repeated_and_a_changed_one_is(self, settings, telemetry, sent, monkeypatch):
        telemetry.feed_count("iCloud Photos", 10)
        telemetry.status()
        telemetry.status()
        telemetry.feed_count("iCloud Photos", 30)
        telemetry.status()
        monkeypatch.setattr(config_module.config, "enabled_integrations", lambda: ["Tesla", "Weather", "Ring"])
        telemetry.status()
        telemetry.stop()
        statuses = _by_name(sent, "status")
        assert [s["properties"]["feeds"]["iCloud Photos"] for s in statuses] == [10, 30, 30]
        assert statuses[-1]["properties"]["integrations"] == ["Ring", "Tesla", "Weather"]

    def test_the_worker_sends_the_status_on_its_own_after_the_first_delay_and_then_on_the_interval(
            self, settings, sent, monkeypatch):
        monkeypatch.setattr(telemetry_module, "STATUS_FIRST_DELAY", 0.005)
        monkeypatch.setattr(telemetry_module, "STATUS_INTERVAL", 0.005)
        t = Telemetry(post=sent.append, install_age_days=lambda: 2.0)
        t.start()
        t.feed_count("iMessage", 400)

        def reported() -> list[dict]:
            return [s["properties"]["feeds"] for s in _by_name(sent, "status")]

        assert wait_until(lambda: {"iMessage": 400} in reported())
        t.feed_count("iMessage", 410)
        assert wait_until(lambda: {"iMessage": 410} in reported())
        t.stop()
        feeds = reported()
        assert feeds.index({"iMessage": 400}) < feeds.index({"iMessage": 410})

    def test_a_status_sizes_the_prompt_as_the_dashboard_does(self, settings, sent):
        stats = [("memory", 2048, 0.0), ("weather", 512, 0.0), ("history", 1024, 0.0), ("static", 3072, 0.0)]
        t = Telemetry(post=sent.append, install_age_days=lambda: 2.0)
        t.start(prompt_stats=lambda: stats)
        t.status()
        t.stop()
        [status] = _by_name(sent, "status")
        assert status["properties"]["prompt_kb"] == 6.5
        assert status["properties"]["prompt_sections"] == {"memory": 2.0, "weather": 0.5, "history": 1.0, "static": 3.0}

    def test_a_prompt_not_yet_built_or_a_failing_provider_leaves_the_status_without_a_size(self, settings, sent):
        stats = {"value": []}

        def provider():
            if stats["value"] is None:
                raise RuntimeError("no prompt manager")
            return stats["value"]

        t = Telemetry(post=sent.append, install_age_days=lambda: 2.0)
        t.start(prompt_stats=provider)
        t.feed_count("iMessage", 1)
        t.status()
        stats["value"] = None
        t.feed_count("iMessage", 2)
        t.status()
        stats["value"] = [("static", 1024, 0.0)]
        t.status()
        t.stop()
        statuses = _by_name(sent, "status")
        assert [s["properties"].get("prompt_kb") for s in statuses] == [None, None, 1.0]
        assert all("prompt_sections" not in s["properties"] for s in statuses[:2])

    def test_an_install_that_opted_out_sends_no_status_or_memory_size(self, settings, telemetry, sent):
        settings["telemetry.enabled"] = False
        telemetry.feed_count("iMessage", 400)
        telemetry.status()
        telemetry.memory_size(3000, 19000, 4, 7)
        telemetry.stop()
        assert not _by_name(sent, "status") and not _by_name(sent, "memory_size")


class TestTranscriptWindow:
    def test_the_window_closes_on_the_seventh_day_without_touching_config(self, settings, sent):
        age = [6.9]
        t = Telemetry(post=sent.append, install_age_days=lambda: age[0])
        t.start()
        t.user_turn("hi", "console")
        age[0] = 7.0
        t.user_turn("what did Alice write?", "console")
        t.stop()
        first, second = _by_name(sent, "user_turn")
        assert first["properties"]["query"] == "hi"
        assert "query" not in second["properties"]
        assert [c.args[0] for c in ConfigStore.set_value.call_args_list] == ["telemetry.id"]

    def test_an_install_without_a_first_run_state_sends_no_text(self, settings, sent):
        t = Telemetry(post=sent.append, install_age_days=lambda: None)
        t.start()
        t.user_turn("hi", "console")
        t.stop()
        [turn] = _by_name(sent, "user_turn")
        assert "query" not in turn["properties"]
        [heartbeat] = _by_name(sent, "heartbeat")
        assert heartbeat["properties"]["content"] is False

    def test_a_content_false_left_by_the_old_latch_in_config_does_not_close_the_window(self, settings, sent):
        """Contract with the beta installs' config.yaml: the earlier rule latched
        `telemetry.content: false` there, and a two-day-old install must get its remaining days."""
        settings["telemetry.content"] = False
        t = Telemetry(post=sent.append, install_age_days=lambda: 2.0)
        t.start()
        t.user_turn("hi", "console")
        t.stop()
        [turn] = _by_name(sent, "user_turn")
        assert turn["properties"]["query"] == "hi"

    def test_the_default_age_is_the_first_run_state_s(self, monkeypatch):
        from assistant import FirstRun
        monkeypatch.setattr(FirstRun, "install_age_days", lambda: None)
        assert telemetry_module.install_age_days() is None
        monkeypatch.setattr(FirstRun, "install_age_days", lambda: 3.5)
        assert telemetry_module.install_age_days() == 3.5


class TestConfigContract:
    def test_the_telemetry_keys_are_settable_through_the_config_store(self):
        """Contract with config.example.yaml: the server writes `telemetry.id` through
        ConfigStore, which refuses keys the example lacks."""
        ConfigStore.check_key("telemetry.id")


def _fail_at_the_garage():
    raise ValueError("no answer from the garage door")


class DoorAway(Exception):
    """A failure the garage code named: defined in the tree, as far as its module says."""
    __module__ = "integrations.garage.GarageClient"


class TestErrorReports:
    """An `error()` the server logs goes out as PostHog's `$exception` event when it may be a
    defect, and as a plain `problem` event when the code foresaw it."""

    def _report(self, telemetry, message="Garage: the door did not answer", raiser=_fail_at_the_garage):
        try:
            raiser()
        except Exception as e:
            telemetry.error_report(message, e)

    def test_a_logged_error_reaches_posthog_through_the_logger(self, telemetry, sent):
        from common.logging import error
        try:
            _fail_at_the_garage()
        except ValueError:
            error("Garage: the door did not answer")
        telemetry.stop()
        [event] = _by_name(sent, "$exception")
        [entry] = event["properties"]["$exception_list"]
        assert entry["type"] == "ValueError" and "door did not answer" in entry["value"]
        assert entry["mechanism"] == {"handled": True, "synthetic": False}

    def test_the_stack_names_the_raising_function_inside_the_checkout(self, telemetry, sent):
        self._report(telemetry)
        telemetry.stop()
        frames = _by_name(sent, "$exception")[0]["properties"]["$exception_list"][0]["stacktrace"]["frames"]
        assert frames[-1]["function"] == "_fail_at_the_garage" and frames[-1]["in_app"]
        assert frames[-1]["filename"] == "tests/unit/assistant/test_telemetry.py"

    def test_no_frame_says_where_the_install_lives(self, telemetry, sent):
        def in_a_library():
            import json
            json.loads("{")
        self._report(telemetry, raiser=in_a_library)
        telemetry.stop()
        frames = _by_name(sent, "$exception")[0]["properties"]["$exception_list"][0]["stacktrace"]["frames"]
        assert not any(f["filename"].startswith("/") for f in frames)
        assert not frames[-1]["in_app"]

    def test_an_error_without_an_exception_is_placed_by_its_caller(self, telemetry, sent):
        def notices_a_problem():
            telemetry.error_report("Feed: the cursor went backwards")
        notices_a_problem()
        telemetry.stop()
        [event] = _by_name(sent, "problem")
        assert event["properties"]["type"] == "Error" and event["properties"]["function"] == "notices_a_problem"
        assert event["properties"]["file"] == "tests/unit/assistant/test_telemetry.py"

    def test_a_failure_the_code_named_is_a_problem_and_no_issue_in_the_error_tracking(self, telemetry, sent):
        def the_door_is_away():
            raise DoorAway("no answer from 192.0.2.7")
        self._report(telemetry, message="Garage: m@mlab.test could not be told", raiser=the_door_is_away)
        telemetry.stop()
        assert _by_name(sent, "$exception") == []
        [event] = _by_name(sent, "problem")
        properties = event["properties"]
        assert properties["type"] == "DoorAway" and properties["function"] == "the_door_is_away"
        assert "could not be told" in properties["message"] and "m@mlab.test" not in properties["message"]

    def test_a_problem_names_what_it_came_from(self, telemetry, sent):
        def wrapped():
            try:
                raise ConnectionResetError("reset by peer")
            except ConnectionResetError as e:
                raise RuntimeError("the door did not answer") from e
        self._report(telemetry, raiser=wrapped)
        telemetry.stop()
        [event] = _by_name(sent, "problem")
        assert event["properties"]["type"] == "RuntimeError"
        assert event["properties"]["caused_by"] == ["ConnectionResetError"]

    def test_a_call_site_that_reports_the_outside_files_no_issue_unless_the_code_broke(self, telemetry, sent):
        from common.logging import error

        def a_check_that_fails():
            try:
                raise RuntimeError("Work: token expired")
            except RuntimeError:
                error("Garage: health check failed", outside=True)

        def a_check_that_is_broken():
            try:
                None.state
            except AttributeError:
                error("Garage: health check failed", outside=True)
        a_check_that_fails()
        a_check_that_is_broken()
        telemetry.stop()
        assert [e["properties"]["type"] for e in _by_name(sent, "problem")] == ["RuntimeError"]
        [issue] = _by_name(sent, "$exception")
        assert issue["properties"]["$exception_list"][0]["type"] == "AttributeError"

    def test_a_repeating_problem_is_held_back_like_a_repeating_issue(self, telemetry, sent, monkeypatch):
        def notices():
            telemetry.error_report("Feed: the source is away")
        for _ in range(3):
            notices()
        clock = time.time()
        monkeypatch.setattr(telemetry_module, "time", SimpleNamespace(time=lambda: clock + telemetry_module.ERROR_INTERVAL + 1))
        notices()
        telemetry.stop()
        assert [e["properties"]["repeats"] for e in _by_name(sent, "problem")] == [0, 2]

    def test_an_uncaught_exception_is_marked_unhandled(self, telemetry, sent):
        from common.logging import error
        error("Uncaught exception ended thread feed", ValueError("gone"))
        telemetry.stop()
        [entry] = _by_name(sent, "$exception")[0]["properties"]["$exception_list"]
        assert entry["mechanism"]["handled"] is False

    def test_the_exception_it_came_from_is_reported_behind_it(self, telemetry, sent):
        def wrapped():
            try:
                _fail_at_the_garage()
            except ValueError as e:
                raise RuntimeError("the evening scene failed") from e
        self._report(telemetry, message="Scene: could not run", raiser=wrapped)
        telemetry.stop()
        entries = _by_name(sent, "$exception")[0]["properties"]["$exception_list"]
        assert [e["type"] for e in entries] == ["RuntimeError", "ValueError"]
        assert "garage door" in entries[1]["value"]

    def test_the_message_goes_out_scrubbed(self, telemetry, sent):
        self._report(telemetry, message="Mail: could not send to alice@example.com from +46701740605")
        telemetry.stop()
        value = _by_name(sent, "$exception")[0]["properties"]["$exception_list"][0]["value"]
        assert "alice@example.com" not in value and "46701740605" not in value
        assert value.startswith("Mail: could not send to")

    def test_errors_go_out_past_the_transcript_window(self, settings, sent):
        t = Telemetry(post=sent.append, install_age_days=lambda: 400.0)
        t.start()
        self._report(t)
        t.stop()
        assert len(_by_name(sent, "$exception")) == 1

    def test_a_repeating_error_is_one_event_and_the_next_says_how_many_were_held_back(
            self, telemetry, sent, monkeypatch):
        for _ in range(5):
            self._report(telemetry)
        clock = time.time()
        monkeypatch.setattr(telemetry_module, "time", SimpleNamespace(time=lambda: clock + telemetry_module.ERROR_INTERVAL + 1))
        self._report(telemetry)
        telemetry.stop()
        assert [e["properties"]["repeats"] for e in _by_name(sent, "$exception")] == [0, 4]

    def test_errors_from_different_places_are_not_held_back_by_each_other(self, telemetry, sent):
        def elsewhere():
            raise ValueError("no answer from the boiler")
        self._report(telemetry)
        self._report(telemetry, raiser=elsewhere)
        telemetry.stop()
        assert len(_by_name(sent, "$exception")) == 2

    def test_an_opt_out_stops_the_error_reports(self, telemetry, settings, sent):
        settings["telemetry.enabled"] = False
        self._report(telemetry)
        telemetry.stop()
        assert _by_name(sent, "$exception") == []

    def test_a_stopped_telemetry_no_longer_listens_to_the_logger(self, telemetry, sent):
        import common.logging as logging_module
        telemetry.stop()
        assert logging_module._error_sink is None

    def test_message_only_errors_logged_from_two_places_are_two_events_each_placed_by_its_caller(
            self, telemetry, sent):
        """Through `error()` the frames of the logger and of the telemetry lie between the
        caller and the report: were they not left out, every such error of the install would
        be one place — and one event an hour."""
        from common.logging import error

        def the_feed_notices():
            error("Feed: the cursor went backwards")

        def the_garage_notices():
            error("Garage: the door reports two states")
        the_feed_notices()
        the_garage_notices()
        telemetry.stop()
        assert [e["properties"]["function"] for e in _by_name(sent, "problem")] \
            == ["the_feed_notices", "the_garage_notices"]

    def test_the_text_of_the_exception_behind_goes_out_scrubbed_too(self, telemetry, sent):
        def wrapped():
            try:
                raise ValueError("no such recipient: alice@example.com")
            except ValueError as e:
                raise RuntimeError("the mail was not sent") from e
        self._report(telemetry, message="Mail: could not send", raiser=wrapped)
        telemetry.stop()
        entries = _by_name(sent, "$exception")[0]["properties"]["$exception_list"]
        assert entries[1]["type"] == "ValueError"
        assert "alice@example.com" not in entries[1]["value"] and "no such recipient" in entries[1]["value"]

    def test_a_vault_value_in_an_error_leaves_as_its_placeholder(self, telemetry, sent, monkeypatch):
        monkeypatch.setattr(telemetry_module, "redact_for_storage",
                            lambda text: text.replace("Gate#Frances", "{{secret:gate-code}}"))

        def wrapped():
            try:
                raise ValueError("the lock refused Gate#Frances")
            except ValueError as e:
                raise RuntimeError("gate closed") from e
        self._report(telemetry, message="Gate: Gate#Frances was refused", raiser=wrapped)
        telemetry.stop()
        entries = _by_name(sent, "$exception")[0]["properties"]["$exception_list"]
        assert not any("Gate#Frances" in e["value"] for e in entries)
        assert all("gate-code" in e["value"] for e in entries)

    def test_an_exception_raised_from_none_does_not_bring_the_one_it_hid(self, telemetry, sent):
        def hides_its_cause():
            try:
                _fail_at_the_garage()
            except ValueError:
                raise RuntimeError("the evening scene failed") from None
        self._report(telemetry, raiser=hides_its_cause)
        telemetry.stop()
        entries = _by_name(sent, "$exception")[0]["properties"]["$exception_list"]
        assert [e["type"] for e in entries] == ["RuntimeError"]

    def test_a_long_chain_is_cut_and_a_chain_in_a_circle_ends(self, telemetry, sent):
        def twelve_deep():
            error = None
            for depth in range(12):
                try:
                    raise ValueError(f"layer {depth}") from error
                except ValueError as e:
                    error = e
            raise error

        def in_a_circle():
            first, second = KeyError("zones"), LookupError("rooms")
            first.__cause__, second.__cause__ = second, first
            raise first
        self._report(telemetry, raiser=twelve_deep)
        self._report(telemetry, raiser=in_a_circle)
        telemetry.stop()
        long, circle = [e["properties"]["$exception_list"] for e in _by_name(sent, "$exception")]
        assert len(long) == 1 + telemetry_module.ERROR_CHAIN_LIMIT
        assert "layer 11" not in long[1]["value"] and "layer 10" in long[1]["value"]   # newest first
        assert [e["type"] for e in circle] == ["KeyError", "LookupError"]

    def test_a_failed_post_is_not_reported_as_an_error_of_its_own(self, settings):
        posts = []

        def failing_post(payload):
            posts.append(payload)
            raise ConnectionError("PostHog is away")
        t = Telemetry(post=failing_post, install_age_days=lambda: 2.0)
        t.start()
        t.stop()
        assert posts and not any(e["event"] == "$exception" for p in posts for e in p["batch"])
        assert t._queue.empty()


class TestReactivationEvents:
    """The property names are a contract with the PostHog insights built on them."""

    def test_a_spent_step_goes_out_with_its_outcome_and_no_text(self, telemetry, sent):
        telemetry.reactivation_nudge(step=3, kind="daily", outcome="sent", via=["chat"],
                                     owner_channels=["Telegram", "Console"], inactive_hours=71.96)
        telemetry.stop()

        (event,) = _by_name(sent, "reactivation_nudge")
        properties = {k: v for k, v in event["properties"].items() if k not in ("version", "$lib")}
        assert properties == {"step": 3, "kind": "daily", "outcome": "sent", "via": ["chat"],
                              "owner_channels": ["Telegram", "Console"], "inactive_hours": 72.0}

    def test_a_return_after_messages_says_how_long_after_the_last_one(self, telemetry, sent):
        telemetry.reactivation_return(inactive_hours=78.04, nudges_sent=2, last_step=3, hours_since_nudge=6.02,
                                      source="Telegram")
        telemetry.stop()

        (event,) = _by_name(sent, "reactivation_return")
        properties = {k: v for k, v in event["properties"].items() if k not in ("version", "$lib")}
        assert properties == {"source": "Telegram", "inactive_hours": 78.0, "nudges_sent": 2, "last_step": 3,
                              "hours_since_nudge": 6.0}

    def test_a_return_nobody_was_written_to_for_carries_no_step(self, telemetry, sent):
        telemetry.reactivation_return(inactive_hours=30, nudges_sent=0, last_step=None, hours_since_nudge=None,
                                      source="Console")
        telemetry.stop()

        (event,) = _by_name(sent, "reactivation_return")
        assert event["properties"]["nudges_sent"] == 0
        assert "last_step" not in event["properties"] and "hours_since_nudge" not in event["properties"]

    def test_an_opted_out_install_sends_neither(self, telemetry, settings, sent):
        settings["telemetry.enabled"] = False
        telemetry.reactivation_nudge(step=1, kind="daily", outcome="declined", via=[], owner_channels=[],
                                     inactive_hours=23)
        telemetry.reactivation_return(inactive_hours=30, nudges_sent=0, last_step=None, hours_since_nudge=None,
                                      source="Console")
        telemetry.stop()

        assert not _by_name(sent, "reactivation_nudge") and not _by_name(sent, "reactivation_return")
