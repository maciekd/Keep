"""The TUI console interface: the ops dashboard with the Home tab chat, served over TCP to
`tui.py`. The chat is a `ConsoleTransport` on the process-wide ChatHub, so the console is
an owner conversation like Telegram (Notifier broadcasts, approval prompts) and its replies
run through the shared `ChatSession`; a `HistoryFeed` fills the same pane with the day's
dialog history. Starts before voice so early log lines show; `attach_voice()` fills the
Developer ▸ Voice tab and outpost shortcuts in once the voice interface is up."""
from common.ConfigManager import config
from common.logging import info
from interfaces.Interface import Interface, bind_host

DEFAULT_TUI_PORT = 5151


class ConsoleInterface(Interface):
    """Starts the TUI ops console, registers its chat transport on the hub and serves the
    console over TCP to `tui.py` clients."""

    name = "console"

    def __init__(self):
        self.console = None
        self.server = None
        self.transport = None
        self.feed = None
        self.hub = None

    def start(self, assistant, app) -> None:
        from assistant.Notifier import set_notifier
        from interfaces.chat.ChatHub import ChatHub
        from interfaces.console.ConsoleTransport import ConsoleTransport
        from interfaces.console.HistoryFeed import HistoryFeed
        from interfaces.console.framework.TUIServer import TUIServer
        from interfaces.console.TUIFloxyConsole import TUIFloxyConsole
        from interfaces.console.widgets.ChatPane import ChatPane

        pane = ChatPane()
        self.feed = HistoryFeed(assistant.prompt_manager.message_history, pane)
        self.transport = ConsoleTransport(pane, self.feed)
        self.console = TUIFloxyConsole.get_instance(None, None, assistant.prompt_manager, pane)
        assistant.subscribe(self.console)
        assistant.subscribe(self.feed)
        self.feed.start()

        self.hub = ChatHub.get_instance()
        self.hub.attach(assistant)
        self.hub.register(self.transport)
        set_notifier(self.hub)
        self.transport.start(self.hub)

        port = int(config.get('interfaces.console.port', DEFAULT_TUI_PORT))
        self.server = TUIServer(self.console.draw_screen, input_callback=self.console.input_callback,
                                host=bind_host(config, "The TUI console"), port=port)
        self.transport.has_clients = lambda: bool(self.server.clients)
        self.server.start()
        info(f"Console interface started (TUI server on port {port}, chat transport on the hub).")

    def after_intro(self, fn) -> None:
        """Run `fn` once the startup intro has played for the first TUI client (at once when
        it has): the launcher's first-run brief waits for the chat it is answered in."""
        self.console.intro.on_done(fn)

    def at_intro_start(self, fn) -> None:
        """Run `fn` as the startup intro starts for the first TUI client (at once when it
        has): a turn that opens the conversation on a local model prefills behind it."""
        self.console.intro.on_started(fn)

    def attach_voice(self, voice) -> None:
        self.console.attach_voice(voice.speech_analyzer, voice.outpost_server)

    def stop(self) -> None:
        if self.feed is not None:
            self.feed.stop()
        if self.transport is not None and self.hub is not None:
            self.hub.unregister(self.transport)
            self.transport.stop()
        if self.server is not None:
            self.server.stop()
