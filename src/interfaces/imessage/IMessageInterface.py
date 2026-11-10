"""The iMessage interface: the assistant as a contact of its own on iMessage, a chat transport
on the process-wide ChatHub over the Messages account of the assistant's Apple ID. Off unless
`interfaces.imessage.enabled` is true; without an allowlist (or, for a satellite, a token) it
still starts, as an "iMessage Chat" integration the monitor lists as not configured, so the
assistant can have the keys typed (config_request) and bring the chat up with
`restart_integration("iMessage Chat")` — no server restart."""
from common.logging import info
from interfaces.Interface import Interface


class IMessageInterface(Interface):
    """Starts the iMessage transport on the ChatHub, which it installs as the process-wide Notifier."""

    name = "imessage"
    enabled_by_default = False

    def __init__(self):
        self.transport = None
        self.hub = None

    def start(self, assistant, app) -> None:
        from assistant.integration.IntegrationMonitor import IntegrationMonitor
        from assistant.Notifier import set_notifier
        from interfaces.chat.ChatHub import ChatHub
        from interfaces.imessage.IMessageTransport import IMessageTransport

        self.hub = ChatHub.get_instance()
        self.hub.attach(assistant)
        set_notifier(self.hub)
        # By class, so the monitor can rebuild it: `_start_transport` runs now and on every
        # restart_integration of it, as for the Telegram bot.
        IntegrationMonitor.get_instance().register_class(IMessageTransport, start=self._start_transport)
        info("iMessage interface started.")

    def _start_transport(self, transport) -> None:
        if self.transport is not None and self.transport is not transport:
            self.hub.unregister(self.transport)     # the rebuilt transport replaces the old one on the hub
        self.transport = transport
        if not transport.is_configured:
            info(f"iMessage chat is not configured yet ({', '.join(transport.missing_config_keys())}) — listed as "
                 f'not configured, restart_integration("{transport.name}") starts it once the keys are set.')
            return
        self.hub.register(transport)
        transport.start(self.hub)

    def stop(self) -> None:
        if self.transport is not None:
            try:
                self.hub.unregister(self.transport)
                self.transport.stop()
            except Exception:
                pass
