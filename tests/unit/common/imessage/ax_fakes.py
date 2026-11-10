"""Messages.app as the Accessibility API shows it, in memory: a main window with a sidebar of
chat rows, and a window of its own for each chat a row is asked to open. `FakeMessages` stands
where `common.imessage.Accessibility` does, so the typing dots are tested without a Mac — and
keeps what a phone would have seen: every value that was written into a compose field."""
from typing import Callable, Optional

from common.imessage.Accessibility import Session
from common.imessage.TypingIndicator import (CHILDREN, CLOSE_BUTTON, COMPOSE_FIELD, CONVERSATION_LIST, DESCRIPTION,
                                             IDENTIFIER, MESSAGES_BUNDLE, PRESS, ROLE, ROW_ROLE, TITLE, VALUE, WINDOWS)

OPEN = "Name:Open in New Window\nTarget:0x0\nSelector:(null)"      # a custom action as the API names it


class Node:
    """An element: its attributes, its children, what its actions do."""

    def __init__(self, role: str = "AXGroup", identifier: Optional[str] = None, title: Optional[str] = None,
                 description: Optional[str] = None, value: Optional[str] = None, children=(), actions=None):
        self.attributes = {ROLE: role, IDENTIFIER: identifier, TITLE: title, DESCRIPTION: description, VALUE: value}
        self.children = list(children)
        self.actions: dict[str, Callable[[], None]] = dict(actions or {})
        self.dead = False           # a closed window's elements no longer answer
        self.frozen = False         # a field that takes no value

    def all(self):
        yield self
        for child in self.children:
            yield from child.all()


class FakeMessages:
    """The seam's calls over a tree of `Node`s. The switches are the gates; `chat()` puts a
    conversation in the sidebar; `typed` is every (window title, value) a field was given."""

    def __init__(self):
        self.granted = True
        self.session_state: Optional[Session] = Session(on_console=True, locked=False)
        self.running = True
        self.sidebar = Node(identifier=CONVERSATION_LIST)
        self.main = self._window("Messages", [Node(children=[self.sidebar]), self._field()])
        self.app = Node("AXApplication", children=[self.main])
        self.windows = [self.main]
        self.typed: list[tuple[str, str]] = []
        self.opened: list[str] = []         # the titles of the windows rows opened, in order
        self.closed: list[str] = []
        self.calls = 0                      # every call of the seam: what a gate must keep at zero
        self.pending: list[Node] = []       # windows a press opened that are not on screen yet
        self.open_after_polls = 0           # how many looks at the window list an opening window takes
        self._polls_left = 0
        self.on_call: Optional[Callable[[], None]] = None

    # --- building the tree -------------------------------------------------------------------

    @staticmethod
    def _field() -> Node:
        return Node("AXTextArea", COMPOSE_FIELD, value="")

    def _window(self, title: str, children) -> Node:
        window = Node("AXWindow", title=title, children=children)
        button = Node("AXButton", actions={PRESS: lambda: self._close(window)})
        window.attributes[CLOSE_BUTTON] = button
        return window

    def chat(self, name: str, latest: str = "See you at seven", opens: Optional[str] = None,
             labelled: bool = False) -> Node:
        """A conversation in the sidebar. Its row opens a window titled `opens` (the chat's name
        unless said). `labelled`: the row has the name once more as a label inside it."""
        row = Node(ROW_ROLE, description=f"{name}, {latest}")
        row.actions[OPEN] = lambda: self._open(name if opens is None else opens)
        if labelled:
            row.children.append(Node(ROW_ROLE, description=name))
        self.sidebar.children.append(row)
        return row

    def chat_window(self, title: str) -> Node:
        """A chat's own window as it is once open — also one the user opened themselves."""
        window = self._window(title, [Node(children=[Node("AXScrollArea"), self._field()])])
        self.windows.append(window)
        return window

    def _open(self, title: str):
        window = self._window(title, [Node(children=[Node("AXScrollArea"), self._field()])])
        self.opened.append(title)
        self.pending.append(window)
        self._polls_left = self.open_after_polls

    def _close(self, window: Node):
        self.closed.append(window.attributes[TITLE])
        if window in self.windows:
            self.windows.remove(window)
        for node in window.all():
            node.dead = True

    def window(self, title: str) -> Optional[Node]:
        return next((w for w in self.windows if w.attributes[TITLE] == title), None)

    def field_of(self, title: str) -> Node:
        return next(node for node in self.window(title).all() if node.attributes[IDENTIFIER] == COMPOSE_FIELD)

    def draft(self, title: str) -> str:
        return self.field_of(title).attributes[VALUE]

    # --- the seam ------------------------------------------------------------------------------

    def _called(self):
        self.calls += 1
        if self.on_call is not None:
            self.on_call()

    def trusted(self) -> bool:
        self._called()
        return self.granted

    def session(self) -> Optional[Session]:
        self._called()
        return self.session_state

    def application(self, bundle_id: str):
        self._called()
        return self.app if self.running and bundle_id == MESSAGES_BUNDLE else None

    def attribute(self, element, name: str):
        self._called()
        if element.dead:
            return None
        if element is self.app and name == WINDOWS:
            if self.pending:
                if self._polls_left <= 0:
                    self.windows.extend(self.pending)
                    self.pending = []
                self._polls_left -= 1
            return list(self.windows)
        if name == CHILDREN:
            return list(element.children)
        return element.attributes.get(name)

    def set_attribute(self, element, name: str, value: str) -> bool:
        self._called()
        if element.dead or element.frozen:
            return False
        element.attributes[name] = value
        owner = next((w for w in self.windows if element in w.all()), None)
        self.typed.append((owner.attributes[TITLE] if owner else "?", value))
        return True

    def actions(self, element) -> list[str]:
        self._called()
        return [] if element.dead else list(element.actions)

    def perform(self, element, action: str) -> bool:
        self._called()
        if element.dead or action not in element.actions:
            return False
        element.actions[action]()
        return True

    def same(self, first, second) -> bool:
        return first is second


class Clock:
    """Time that moves only when told, or when the code under test sleeps."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Timers:
    """Stands where `threading.Timer` does: the keep-alive is run by the test, one tick at a time."""

    def __init__(self, clock: Clock):
        self.clock = clock
        self.waiting: list["Timers.Timer"] = []

    class Timer:
        def __init__(self, delay, call):
            self.delay, self.call, self.cancelled = delay, call, False

        def cancel(self):
            self.cancelled = True

    def __call__(self, delay: float, call) -> "Timers.Timer":
        timer = self.Timer(delay, call)
        self.waiting.append(timer)
        return timer

    @property
    def armed(self) -> bool:
        return any(not timer.cancelled for timer in self.waiting)

    def tick(self) -> bool:
        """Let the next timer's time pass and run it; False when none was waiting."""
        live = [timer for timer in self.waiting if not timer.cancelled]
        self.waiting = []
        if not live:
            return False
        self.clock.now += live[0].delay
        live[0].call()
        return True
