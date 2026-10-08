import threading

from assistant.commands.CommandSet import CommandSet
from assistant.commands.Commands import normalize_llm_escapes
from assistant.SystemNote import system_note
from common.logging import info, trace
from integrations.whatsapp.WhatsAppMessages import SERVER_GROUP, jid_parts

NOTE_SOURCE = "WhatsApp"
INTEGRATION = "WhatsApp"
LINK_STEPS = ("Open WhatsApp on the phone: Settings (or the ⋮ menu) > Linked devices > Link a device > "
              "\"Link with phone number instead\", and type the code there. It is valid for a few minutes.")


class WhatsAppCommands(CommandSet):
    """Links the WhatsApp account from the chat and sends a WhatsApp message (sensitive, so it
    passes the approval gate first)."""
    MULTI_PARAM_COMMANDS = frozenset({"send_whatsapp"})
    SENSITIVE_COMMANDS = frozenset({"send_whatsapp"})

    # --- whatsapp_login ---------------------------------------------------------------------------

    def cmd_whatsapp_login(self, parameter_string: str):
        """Asks WhatsApp for a pairing code for the owner's number and hands it back for the
        user to type into the phone; a system note says when the link landed. Works from any
        chat: the code goes into the phone, nothing secret comes back here."""
        phone = parameter_string.strip().strip('"\'').replace(" ", "")
        if not phone.startswith("+") or not phone[1:].isdigit() or len(phone) < 8:
            return ('whatsapp_login failed: expected the phone number with its country code, '
                    'e.g. whatsapp_login("+46701234567")')
        self.background_command(self._run_whatsapp_login, phone, lane=self.commands.CONFIG_LANE)

    def _run_whatsapp_login(self, phone: str) -> str:
        monitor = self.commands.integration_monitor()
        integration = monitor.find(INTEGRATION)
        if integration is None or not hasattr(integration, "login"):
            state = getattr(integration, "state", None)
            label = self.commands.state_label(getattr(state, "value", state), "not available")
            info(f"whatsapp_login: WhatsApp is {label}, no login possible")
            if label == "disabled":
                return f'WhatsApp is disabled — enable_integration("{INTEGRATION}") first.'
            return f"WhatsApp is {label} — enable it first."
        code, failure = integration.login(phone)
        if failure:
            info(f"whatsapp_login: {failure}")
            return failure
        threading.Thread(target=self._await_pairing, args=(integration, phone), name="WhatsAppPairing",
                         daemon=True).start()
        return (f"WhatsApp pairing code for {phone}: {code}. {LINK_STEPS} A system note follows once the phone "
                f"has accepted it and the feed has started.")

    def _await_pairing(self, integration, phone: str):
        failure = integration.wait_paired()
        if failure:
            info(f"whatsapp_login: pairing for {phone} did not complete: {failure}")
            system_note(NOTE_SOURCE, f"WhatsApp link for {phone} did not complete: {failure}")
            return
        result = integration.finish_login()
        status = (result or {}).get("status") or {}
        state = self.commands.state_label(status.get("state"), "unknown")
        trace(f"whatsapp_login: paired, feed started, state {state}")
        system_note(NOTE_SOURCE, f"WhatsApp is linked for {phone}; the feed is {state} and the history is on its way.")

    def describe_whatsapp_login(self, p):     return "Linking the WhatsApp account"

    # --- send_whatsapp ----------------------------------------------------------------------------

    def cmd_send_whatsapp(self, *args: str) -> str:
        if len(args) < 2:
            return "send_whatsapp failed: expected 'recipient, message' parameters"
        recipient = args[0].strip()
        # An unquoted body arrives split on its commas; re-join it.
        body = ", ".join(a for a in args[1:]).strip()
        if not recipient or not body:
            return "send_whatsapp failed: expected 'recipient, message' parameters"
        body = normalize_llm_escapes(body)
        body = self.commands.with_ai_footer(body)
        from integrations.whatsapp.WhatsAppIntegration import WhatsAppIntegration
        return WhatsAppIntegration.get_instance().send(recipient, body)

    # The approval card names the number the message goes to, or why there is none, and says
    # when the owner has never written with it — the user approves a number, never a bare name.
    def describe_send_whatsapp(self, p):
        recipient = p.split(",", 1)[0].strip().strip('"\'') if p else ""
        if not recipient:
            return "Sending WhatsApp message"
        try:
            from integrations.whatsapp.WhatsAppIntegration import WhatsAppIntegration
            whatsapp = WhatsAppIntegration.get_instance()
            jid, display, err = whatsapp.resolve_recipient(recipient)
            if err:
                return f"Sending WhatsApp message to {recipient} — {err}"
            number = whatsapp.phone_for(jid)
            if jid_parts(jid)[1] == SERVER_GROUP:
                line = f"Sending WhatsApp message to the group {display}"
            elif number is None or display == number:
                line = f"Sending WhatsApp message to {display}"
            else:
                line = f"Sending WhatsApp message to {display} ({number})"
            if not whatsapp.is_known(recipient):
                line += " — no earlier chat with this number"
            return line
        except Exception as e:
            trace(f"describe_send_whatsapp: lookup failed for '{recipient}': {e}")
        return f"Sending WhatsApp message to {recipient}"
