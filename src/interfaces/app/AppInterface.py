"""The phone app's chat: an `AppTransport` on the process-wide ChatHub and the `/app` routes
the apps speak, on the HTTP interface's Quart app. Enabled by `interfaces.app.enabled` (on by
default): it only adds routes, and nothing reaches them from another machine without
`interfaces.lan_access`."""
from common.logging import info
from common.Paths import CACHE_DIR
from interfaces.Interface import Interface


class AppInterface(Interface):
    """Registers the app chat's transport on the hub and its routes on the HTTP app."""

    name = "app"

    def __init__(self):
        self.transport = None
        self.hub = None

    def start(self, assistant, app) -> None:
        from assistant.Notifier import set_notifier
        from interfaces.app import routes
        from interfaces.app.AppTransport import AppTransport
        from interfaces.app.Transcript import Transcript
        from interfaces.chat.ChatHub import ChatHub

        self.transport = AppTransport(Transcript(CACHE_DIR))
        self.hub = ChatHub.get_instance()
        self.hub.attach(assistant)
        self.hub.register(self.transport)
        set_notifier(self.hub)
        self.transport.start(self.hub)
        routes.register(app, self.transport)
        info("App interface started (chat transport on the hub, routes under /app).")

    def stop(self) -> None:
        if self.transport is not None and self.hub is not None:
            self.hub.unregister(self.transport)
            self.transport.stop()
