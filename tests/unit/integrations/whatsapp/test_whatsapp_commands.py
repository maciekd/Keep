from unittest.mock import Mock, patch

from assistant.commands.Commands import Commands
from assistant.integration.Integration import IntegrationState


def _monitor(integration):
    monitor = Mock()
    monitor.find.return_value = integration
    return monitor


class TestCmdWhatsappLogin:
    def test_runs_in_the_background_from_any_chat(self, commands_instance):
        commands_instance._current_context = Mock(source="Telegram", conversation=123)
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("whatsapp_login", ['"+46 70 174 06 05"']) == ""
        bg.assert_called_once_with(commands_instance._owner("whatsapp_login")._run_whatsapp_login, "+46701740605",
                                   lane=Commands.CONFIG_LANE)

    def test_phone_needs_a_country_code(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            for bad in ("0701740605", "+46", "", "+46abc"):
                assert commands_instance.run("whatsapp_login", [bad]).startswith(
                    "whatsapp_login failed: expected the phone number with its country code")
        bg.assert_not_called()

    def test_run_hands_the_code_back_and_notes_the_link_later(self, commands_instance):
        integration = Mock()
        integration.login.return_value = ("ABCD-EFGH", None)
        integration.wait_paired.return_value = None
        integration.finish_login.return_value = {"status": {"state": "ok"}}
        owner = commands_instance._owner("whatsapp_login")
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(integration)), \
             patch("integrations.whatsapp.WhatsAppCommands.threading.Thread") as thread, \
             patch("integrations.whatsapp.WhatsAppCommands.system_note") as note:
            reply = owner._run_whatsapp_login("+46701740605")
            assert "ABCD-EFGH" in reply and "Linked devices" in reply
            thread.return_value.start.assert_called_once()
            owner._await_pairing(integration, "+46701740605")
        integration.login.assert_called_once_with("+46701740605")
        integration.finish_login.assert_called_once_with()
        assert "linked" in note.call_args.args[1] and "ok" in note.call_args.args[1]

    def test_a_code_not_entered_in_time_is_a_note_without_a_feed_start(self, commands_instance):
        integration = Mock()
        integration.wait_paired.return_value = "the code was not entered in time"
        with patch("integrations.whatsapp.WhatsAppCommands.system_note") as note:
            commands_instance._owner("whatsapp_login")._await_pairing(integration, "+46701740605")
        integration.finish_login.assert_not_called()
        assert "not entered in time" in note.call_args.args[1]

    def test_run_reports_a_refused_login(self, commands_instance):
        integration = Mock()
        integration.login.return_value = (None, "WhatsApp login failed: another login is in progress")
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(integration)):
            assert commands_instance._owner("whatsapp_login")._run_whatsapp_login("+46701740605") == \
                "WhatsApp login failed: another login is in progress"

    def test_run_on_a_disabled_or_missing_integration(self, commands_instance):
        placeholder = Mock(spec=["state"])
        placeholder.state = IntegrationState.DISABLED
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(placeholder)):
            assert commands_instance._owner("whatsapp_login")._run_whatsapp_login("+46701740605") == \
                'WhatsApp is disabled — enable_integration("WhatsApp") first.'
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(None)):
            assert "not available" in commands_instance._owner("whatsapp_login")._run_whatsapp_login("+46701740605")


class TestCmdSendWhatsapp:
    def test_sends_with_the_ai_suffix(self, commands_instance):
        with patch("integrations.whatsapp.WhatsAppIntegration.WhatsAppIntegration") as cls:
            cls.get_instance.return_value.send.return_value = ""
            assert commands_instance.run("send_whatsapp", ["Charlotte", "On my way home"]) == ""
        cls.get_instance.return_value.send.assert_called_once_with(
            "Charlotte", f"On my way home\n\n{commands_instance.AI_MESSAGE_SUFFIX}")

    def test_the_footer_stays_home_when_the_setting_is_off(self, commands_instance):
        import common.ConfigManager as config_module
        real_get = config_module.config.get
        with patch.object(config_module.config, "get",
                          lambda key, *a, **k: False if key == "ai_message_footer" else real_get(key, *a, **k)), \
             patch("integrations.whatsapp.WhatsAppIntegration.WhatsAppIntegration") as cls:
            cls.get_instance.return_value.send.return_value = ""
            commands_instance.run("send_whatsapp", ["Charlotte", "On my way home"])
        cls.get_instance.return_value.send.assert_called_once_with("Charlotte", "On my way home")

    def test_an_unquoted_body_is_rejoined_on_its_commas(self, commands_instance):
        with patch("integrations.whatsapp.WhatsAppIntegration.WhatsAppIntegration") as cls:
            cls.get_instance.return_value.send.return_value = ""
            commands_instance.run("send_whatsapp", ["Bob", "late", "sorry"])
        assert cls.get_instance.return_value.send.call_args.args[1].startswith("late, sorry")

    def test_missing_parts_send_nothing(self, commands_instance):
        with patch("integrations.whatsapp.WhatsAppIntegration.WhatsAppIntegration") as cls:
            assert "send_whatsapp failed" in commands_instance.run("send_whatsapp", ["", "body"])
            assert "send_whatsapp failed" in commands_instance.run("send_whatsapp", ["Bob", "  "])
            cls.get_instance.assert_not_called()

    def test_literal_escapes_become_newlines(self, commands_instance):
        with patch("integrations.whatsapp.WhatsAppIntegration.WhatsAppIntegration") as cls:
            cls.get_instance.return_value.send.return_value = ""
            commands_instance.run("send_whatsapp", ["Bob", "line one\\nline two"])
        assert "line one\nline two" in cls.get_instance.return_value.send.call_args.args[1]

    def test_the_approval_card_shows_the_resolved_number_and_a_stranger_warning(self, commands_instance):
        owner = commands_instance._owner("send_whatsapp")
        with patch("integrations.whatsapp.WhatsAppIntegration.WhatsAppIntegration") as cls:
            wa = cls.get_instance.return_value
            wa.resolve_recipient.return_value = ("46701740611@s.whatsapp.net", "Bob Berg", None)
            wa.phone_for.return_value = "+46701740611"
            wa.is_known.return_value = True
            text = owner.describe_send_whatsapp('"Bob", "hi"')
            assert "Bob Berg" in text and "+46701740611" in text, "a name is approved with the number it resolved to"
            wa.resolve_recipient.return_value = ("46701740699@s.whatsapp.net", "+46701740699", None)
            wa.phone_for.return_value = "+46701740699"
            wa.is_known.return_value = False
            text = owner.describe_send_whatsapp("+46701740699, hi")
            assert text.count("+46701740699") == 1 and "no earlier chat" in text
            wa.resolve_recipient.return_value = ("120363000000000001@g.us", "Family", None)
            wa.phone_for.return_value = None
            wa.is_known.return_value = True
            text = owner.describe_send_whatsapp("Family, hi")
            assert "group" in text and "Family" in text and "+" not in text

    def test_the_approval_card_never_shows_a_lid_as_a_number(self, commands_instance):
        owner = commands_instance._owner("send_whatsapp")
        with patch("integrations.whatsapp.WhatsAppIntegration.WhatsAppIntegration") as cls:
            wa = cls.get_instance.return_value
            wa.resolve_recipient.return_value = ("221010443927802@lid", "Bob Berg", None)
            wa.is_known.return_value = True
            wa.phone_for.return_value = "+46701740611"
            text = owner.describe_send_whatsapp("Bob, hi")
            assert "+46701740611" in text and "221010443927802" not in text
            wa.phone_for.return_value = None
            text = owner.describe_send_whatsapp("Bob, hi")
            assert "Bob Berg" in text and "221010443927802" not in text and "+" not in text
            wa.resolve_recipient.return_value = (None, None, "unknown contact: Carol")
            assert owner.describe_send_whatsapp("Carol, hi") == "Sending WhatsApp message to Carol — unknown contact: Carol"

    def test_the_card_survives_a_lookup_failure(self, commands_instance):
        owner = commands_instance._owner("send_whatsapp")
        with patch("integrations.whatsapp.WhatsAppIntegration.WhatsAppIntegration") as cls:
            cls.get_instance.side_effect = RuntimeError("no client")
            assert owner.describe_send_whatsapp("Bob, hi") == "Sending WhatsApp message to Bob"
