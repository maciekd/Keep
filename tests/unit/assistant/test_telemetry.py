"""The beta telemetry: what leaves the machine, under which switch, and the latch that ends the
transcript the moment a sensitive integration is enabled."""
import uuid
from unittest.mock import Mock

import pytest

import assistant.Telemetry as telemetry_module
import common.ConfigManager as config_module
from assistant.Telemetry import Telemetry, enabled_sensitive_integrations
from common import ConfigSchema, ConfigStore


@pytest.fixture
def settings(monkeypatch):
    """A config with telemetry fully on and no id yet; the project key is built in; config
    writes are recorded, not performed."""
    values = {"telemetry.enabled": True, "telemetry.content": True, "telemetry.id": "", "owner_name": "Marcus"}
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
    t = Telemetry(post=sent.append, sensitive_enabled=lambda: [])
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
        t = Telemetry(post=sent.append, sensitive_enabled=lambda: [])
        t.start()
        t.user_turn("hello", "console")
        t.stop()
        assert not t.active and sent == []

    def test_without_a_built_in_key_nothing_is_sent(self, settings, sent, monkeypatch):
        monkeypatch.setattr(telemetry_module, "POSTHOG_PROJECT_KEY", "")
        t = Telemetry(post=sent.append, sensitive_enabled=lambda: [])
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
        t = Telemetry(post=sent.append, sensitive_enabled=lambda: [])
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

    def test_content_off_in_config_keeps_the_text_out_but_the_command_names_in(self, settings, sent):
        settings["telemetry.content"] = False
        t = Telemetry(post=sent.append, sensitive_enabled=lambda: [])
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
        t = Telemetry(post=sent.append, sensitive_enabled=lambda: [])
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
        import time
        monkeypatch.setattr(telemetry_module, "HEARTBEAT_INTERVAL", 0.002)
        t = Telemetry(post=sent.append, sensitive_enabled=lambda: [])
        t.start()
        deadline = time.time() + 1.0
        while time.time() < deadline and len(_by_name(sent, "heartbeat")) < 3:
            time.sleep(0.002)
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

        t = Telemetry(post=post, sensitive_enabled=lambda: [])
        t.start()
        t.stop()          # the startup heartbeat fails
        t2 = Telemetry(post=post, sensitive_enabled=lambda: [])
        t2.start()
        t2.user_turn("hello", "console")
        t2.stop()
        assert len(posts) >= 2
        assert len(_by_name(posts, "user_turn")) == 1


class TestSensitiveLatch:
    def test_enabling_a_sensitive_integration_latches_content_off_in_config_and_says_which(self, settings, sent):
        enabled = []
        t = Telemetry(post=sent.append, sensitive_enabled=lambda: list(enabled))
        t.start()
        t.user_turn("hi", "console")
        enabled.append("iMessage")
        t.user_turn("what did Alice write?", "console")
        t.stop()
        first, second = _by_name(sent, "user_turn")
        assert first["properties"]["query"] == "hi"
        assert "query" not in second["properties"]
        [off] = _by_name(sent, "content_off")
        assert off["properties"]["integration"] == ["iMessage"]
        assert ("telemetry.content", "false") in [c.args for c in ConfigStore.set_value.call_args_list]

    def test_the_flag_is_read_from_the_enabled_list_not_the_class_switch(self, monkeypatch):
        """The unit conftest stubs `Integration.enabled` to True for everything; the latch
        must look at config.yaml's list, or every test install would latch at once."""
        monkeypatch.setattr(config_module.config, "enabled_integrations", lambda: ["iMessage", "Tesla", "Weather"])
        assert enabled_sensitive_integrations() == ["iMessage"]
        monkeypatch.setattr(config_module.config, "enabled_integrations", lambda: ["Tesla", "Weather"])
        assert enabled_sensitive_integrations() == []


class TestConfigContract:
    def test_the_telemetry_keys_are_settable_through_the_config_store(self):
        """Contract with config.example.yaml: the server writes `telemetry.id` and latches
        `telemetry.content` through ConfigStore, which refuses keys the example lacks."""
        ConfigStore.check_key("telemetry.id")
        ConfigStore.check_key("telemetry.content")
        assert ConfigSchema.leaf_type("telemetry.content") is bool
