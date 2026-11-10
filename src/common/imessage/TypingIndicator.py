"""The typing dots: while a reply is worked on, the people of a chat see the account typing.

Messages shows the dots when its compose field holds text, and the field's value can be set
through Accessibility — no keystroke, no focus, Messages not in front. Each chat gets a window
of its own for that (the "Open in New Window" action of its sidebar row), so two chats can show
dots at once and the main window is never typed into. The field is rewritten every few seconds
while the dots are wanted, cleared when they are not, and the window closed again.

Dots in the wrong chat are the one failure that is not acceptable, so every step that is not
exactly as expected ends in no dots: the chat's row is used only when exactly one visible row
could be the chat's (`names_row`), the window that opens must be new and titled with that name
before anything is written and again before every rewrite, and every walk of the tree has a
depth, a node and a time cap. A row's description cannot tell a chat from one with a longer
name that begins the same ("Anna" and the group "Anna, Bob & Carol"), so the title of the
window is what decides; a chat a press showed to have a longer name is remembered, and its row
is not pressed for the shorter name again. Nothing here raises and nothing here is needed for a send. Why dots do
not show is logged once per cause — a gate of the Mac once for all chats, a chat's own reason
once per chat — never on every rewrite.

All of the Mac sits behind `Accessibility`; the clock, the sleep and the timer are arguments,
so the tests run the keep-alive and the timeout without waiting."""
import atexit
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Optional

from common.imessage.Accessibility import Accessibility
from common.imessage.MessagesHost import (TYPING_BUSY, TYPING_FAILED, TYPING_NO_ACCESSIBILITY, TYPING_NO_MESSAGES,
                                          TYPING_NO_ROW, TYPING_NO_WINDOW, TYPING_NOT_ON_CONSOLE, TYPING_SCREEN_LOCKED,
                                          TYPING_UNAVAILABLE, TypingResult)
from common.logging import error, info, trace

MESSAGES_BUNDLE = "com.apple.MobileSMS"
CONVERSATION_LIST = "ConversationList"      # AXIdentifier of the sidebar
COMPOSE_FIELD = "messageBodyField"          # AXIdentifier of a window's compose field
OPEN_ACTION = "Open in New Window"          # in the name of a sidebar row's custom action
ROW_ROLE = "AXStaticText"
WINDOWS, CHILDREN, IDENTIFIER, ROLE = "AXWindows", "AXChildren", "AXIdentifier", "AXRole"
TITLE, DESCRIPTION, VALUE, CLOSE_BUTTON, PRESS = "AXTitle", "AXDescription", "AXValue", "AXCloseButton", "AXPress"

# What the field holds while the dots show: two values, so each rewrite is a change.
VALUES = ("…", "……")
# The dots were seen to hold at least 10 s after a change (how long exactly is not measured).
KEEPALIVE_SECONDS = 8.0
# Dots nobody turned off stop by themselves: the server died, the network dropped. One `on`
# covers one round of the model and the commands it runs, which is seldom over a minute; three
# cover a slow local model and a long fetch, and a round longer than that loses its dots, not
# its reply.
TIMEOUT_SECONDS = 180.0
MAX_WINDOWS = 4             # chats that show dots at once, each in a window of its own
WINDOW_WAIT = 2.0           # how long a chat's window may take to open (0.3 s when tried)
WINDOW_POLL = 0.1
RETRY_SECONDS = 600.0       # a chat whose row opened another chat's window is left alone this long
KNOWN_LONGER = 64           # names remembered as having a longer-named chat's row, before the memory starts over
MAX_DEPTH = 25              # every walk: a locked session shows a window that contains itself
MAX_NODES = 1500
WALK_SECONDS = 2.0
SCANNED_WINDOWS = 8
SCANNED_COMMAS = 12         # how many commas into a row's description its name may end

_INVISIBLE = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮⁦⁧⁨⁩"))
_PHONE = re.compile(r"^\+\d{7,15}$")
_PHONE_PUNCTUATION = re.compile(r"[ ().\-‐‑]")


def _plain(text) -> str:
    """A name as it is compared: composed, without direction marks, any run of spaces one space."""
    return " ".join(unicodedata.normalize("NFC", str(text or "")).translate(_INVISIBLE).split())


def same_name(shown, name: str) -> bool:
    """Whether what Messages shows is exactly this chat's name. Case and every letter count;
    a phone number may be shown grouped ("+46 70 174 06 05") and is the same number."""
    shown, name = _plain(shown), _plain(name)
    if not name:
        return False
    return shown == name or (bool(_PHONE.match(name)) and _PHONE_PUNCTUATION.sub("", shown) == name)


def names_row(description, name: str) -> bool:
    """Whether a sidebar row could be this chat's. A row's description is the chat's name, a
    comma and the latest message, and both may hold commas: the row could be the chat's when
    the description is the name, or the name up to one of its commas — the whole of what
    stands before that comma, compared as `same_name` does, so "Family Dinner, …" is not
    "Family". That is as much as a description says: "Anna, Bob & Carol, see you" could be
    Anna's row or the row of the group "Anna, Bob & Carol". Which it is shows only in the
    title of the window the row opens."""
    text = _plain(description)
    commas = [at for at, char in enumerate(text) if char == ","][:SCANNED_COMMAS]
    return same_name(text, name) or any(same_name(text[:at], name) for at in commas)


def _daemon_timer(delay: float, call: Callable[[], None]):
    timer = threading.Timer(delay, call)
    timer.daemon = True
    timer.name = "imessage-typing"
    timer.start()
    return timer


@dataclass
class _Window:
    """A chat's own window, opened here."""
    name: str               # what the sidebar calls the chat, and the window's title
    element: Any
    field: Any
    since: float            # when the dots were last asked for
    live: bool = True       # False: the dots are over and the window waits to be cleared and closed
    written: bool = False
    flip: int = 0


@dataclass
class _Look:
    """What one look at the Mac found: the gate that is closed, or Messages as it stands."""
    gate: str = ""
    detail: str = ""
    app: Any = None
    windows: Any = ()
    main: Any = None        # the window with the conversation list
    sidebar: Any = None     # the conversation list itself


class _NotAsExpected(Exception):
    """Something of Messages is not what a step counted on: the step ends without dots."""


class _Capped(_NotAsExpected):
    """A walk reached its depth, node or time cap: what it saw is not the whole tree."""


class TypingIndicator:
    """Shows and removes the typing dots in the chats of this Mac's Messages, one window per chat, and keeps them alive.

    `on(chat, name)` — `name` is what the sidebar of this Mac calls the chat, which the host
    works out — opens the chat's window and writes the field; `off(chat)` clears it and closes
    the window. In between a timer rewrites the field every `KEEPALIVE_SECONDS`, after checking
    the window is still the chat's, and ends dots nobody turned off after `TIMEOUT_SECONDS`.
    Windows that cannot be reached when they should close (the screen was locked meanwhile) are
    closed once they can. `blocked()` is the gate that keeps every chat's dots away, for the
    host's health. A window this object did not open is never written to and never closed."""

    def __init__(self, ax: Optional[Accessibility] = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 timer: Callable[[float, Callable[[], None]], Any] = _daemon_timer):
        self._ax = ax if ax is not None else Accessibility()
        self._clock = clock
        self._sleep = sleep
        self._start_timer = timer
        self._lock = threading.RLock()      # one caller at a time in the tree: chats come from several threads
        self._windows: dict[str, _Window] = {}
        self._timer = None
        self._gate = ""                     # the gate the last look found closed
        self._told: dict[str, str] = {}     # chat → the reason last logged for it
        self._retry_after: dict[str, float] = {}
        self._longer: dict[str, set[str]] = {}      # a name → the longer chat names a press showed its row to be
        self._exit_hooked = False

    # --- the calls ------------------------------------------------------------------------

    def on(self, chat: str, name: str) -> TypingResult:
        with self._lock:
            try:
                return self._on(chat, name)
            except Exception as e:
                return self._gave_up(chat, f"showing the dots failed ({type(e).__name__}: {e})")

    def off(self, chat: str) -> TypingResult:
        with self._lock:
            window = self._windows.get(chat)
            if window is None:
                return TypingResult()
            try:
                gate, detail, _ = self._session_gate()
                if gate:
                    self._wait_for_the_mac(chat, window, detail)
                    return TypingResult(reason=gate, detail=detail)
                self._close(chat, window)
                trace(f"Typing dots [{chat}]: off")
            except Exception as e:
                self._windows.pop(chat, None)
                error(f"Typing dots [{chat}]: taking the dots away failed ({type(e).__name__}: {e}) — "
                      f"a window titled '{window.name}' may be left open with a draft")
            return TypingResult()

    def shutdown(self) -> None:
        """Clear and close every window opened here — the process is ending."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            for chat, window in list(self._windows.items()):
                try:
                    self._close(chat, window)
                except Exception as e:
                    error(f"Typing dots [{chat}]: not cleaned up at shutdown ({type(e).__name__}: {e})")
            self._windows.clear()

    def blocked(self) -> str:
        """The gate that kept the dots away when they were last asked for and still does; ""
        when it is open again, or when nobody asked. Looks at the Mac only while a gate is
        closed, so a host that never shows dots never loads the frameworks for its health."""
        with self._lock:
            if self._gate and self._gate != TYPING_UNAVAILABLE:
                try:
                    self._gates()
                except Exception as e:
                    trace(f"Typing dots: the gates could not be looked at ({type(e).__name__}: {e})")
            return self._gate

    def showing(self) -> list[str]:
        """The chats whose dots are on right now."""
        with self._lock:
            return [chat for chat, window in self._windows.items() if window.live]

    def sidebar(self) -> tuple[list[str], str]:
        """(what each visible sidebar row says — the chat's name, a comma, its latest message —
        "") or ([], why they cannot be read). For the live check a person runs; the dots never
        need the whole list."""
        with self._lock:
            try:
                look = self._gates()
                if look.gate:
                    return [], look.detail
                return [str(self._ax.attribute(row, DESCRIPTION) or "") for row, _ in self._rows(look.sidebar)], ""
            except _Capped as e:
                return [], str(e)
            except Exception as e:
                return [], f"{type(e).__name__}: {e}"

    # --- the gates ------------------------------------------------------------------------

    def _session_gate(self) -> tuple[str, str, Any]:
        """(reason, detail, Messages' root element): whether this process may and can reach
        Messages at all — checked before anything is touched, cheap enough for every call."""
        try:
            trusted = self._ax.trusted()
        except ImportError as e:
            return TYPING_UNAVAILABLE, f"the Accessibility frameworks could not be loaded ({e})", None
        if not trusted:
            return (TYPING_NO_ACCESSIBILITY, "Accessibility is not granted to the app this process runs under "
                    "(Terminal for a Keep started by start.sh): System Settings > Privacy & Security > Accessibility",
                    None)
        session = self._ax.session()
        if session is None or not session.on_console:
            return (TYPING_NOT_ON_CONSOLE, "this macOS user's session is not the one on the Mac's screen (another "
                    "user is switched in, or nobody is logged in at the display)", None)
        if session.locked:
            return TYPING_SCREEN_LOCKED, "the screen is locked, and a locked session shows no window of Messages", None
        app = self._ax.application(MESSAGES_BUNDLE)
        if app is None:
            return TYPING_NO_MESSAGES, "Messages.app is not running", None
        return "", "", app

    def _gates(self) -> _Look:
        """One look at the Mac, down to the sidebar. Its gate is remembered as the one
        `blocked` reports, and logged when it is not what it was."""
        gate, detail, app = self._session_gate()
        look = _Look(gate, detail, app)
        if not gate:
            look.windows = self._app_windows(app)
            look.main, look.sidebar, unread = self._main_window(look.windows)
            if look.sidebar is None:
                look.gate = TYPING_NO_WINDOW
                look.detail = (f"no window of Messages could be read down to its conversation list ({unread})" if unread
                               else "Messages shows no window with its conversation list — open it (Window > "
                                    "Messages) and leave it open")
        if look.gate != self._gate:
            if look.gate:
                info(f"Typing dots: none for any chat — {look.detail}. Replies are sent all the same.")
            else:
                info("Typing dots: Messages can be reached again — dots show from the next reply on")
            self._gate = look.gate
        elif look.gate:
            trace(f"Typing dots: still none — {look.detail}")
        return look

    def _app_windows(self, app) -> list:
        windows = self._ax.attribute(app, WINDOWS)
        return list(windows) if isinstance(windows, list) else []

    def _is_mine(self, window) -> bool:
        return any(self._ax.same(window, own.element) for own in self._windows.values())

    def _main_window(self, windows: list) -> tuple[Any, Any, str]:
        """(the window that has the conversation list, the list, ""), or (None, None, why a
        walk was cut short — "" when none was) when no window has one. A window whose walk
        is cut short has no list as far as anyone can tell."""
        unread = ""
        for window in windows[:SCANNED_WINDOWS]:
            if self._is_mine(window):
                continue
            try:
                found = self._find(window, CONVERSATION_LIST)
            except _Capped as e:
                unread = str(e)
                continue
            if found is not None:
                return window, found, ""
        return None, None, unread

    # --- walking the tree -------------------------------------------------------------------

    def _walk(self, root, what: str):
        """Every element under *root*, nearest first. Ends in `_Capped` at the caps — a tree
        that contains itself, a list longer than anything Messages shows, an app that hangs."""
        started, seen = self._clock(), 0
        level, depth = [root], 0
        while level:
            if depth > MAX_DEPTH:
                raise _Capped(f"{what} is more than {MAX_DEPTH} levels deep")
            following = []
            for element in level:
                seen += 1
                if seen > MAX_NODES:
                    raise _Capped(f"{what} has more than {MAX_NODES} elements")
                if self._clock() - started > WALK_SECONDS:
                    raise _Capped(f"{what} took more than {WALK_SECONDS:.0f} s to read")
                yield element
                children = self._ax.attribute(element, CHILDREN)
                if isinstance(children, list):
                    following.extend(children)
            level, depth = following, depth + 1

    def _find(self, root, identifier: str):
        return next((element for element in self._walk(root, f"the search for {identifier}")
                     if self._ax.attribute(element, IDENTIFIER) == identifier), None)

    def _rows(self, sidebar, name: Optional[str] = None) -> list[tuple[Any, str]]:
        """The visible sidebar rows as (element, its open-in-a-window action) — with a name,
        only the rows that could be that chat's (`names_row`) and are not known to be a
        longer-named chat's. A row is a static text that takes that action; a label inside
        it does not."""
        rows = []
        longer = self._longer.get(_plain(name), ()) if name is not None else ()
        for element in self._walk(sidebar, "the conversation list"):
            if element is sidebar or self._ax.attribute(element, ROLE) != ROW_ROLE:
                continue
            description = self._ax.attribute(element, DESCRIPTION) if name is not None else None
            if name is not None and (not names_row(description, name)
                                     or any(names_row(description, other) for other in longer)):
                continue
            action = next((action for action in self._ax.actions(element) if OPEN_ACTION in action), None)
            if action:
                rows.append((element, action))
        return rows

    # --- on ---------------------------------------------------------------------------------

    def _on(self, chat: str, name: str) -> TypingResult:
        now = self._clock()
        look = self._gates()
        if look.gate:
            return TypingResult(reason=look.gate, detail=look.detail)
        self._sweep()
        window = self._windows.get(chat)
        if window is not None:
            if not self._unexpected(window):
                window.since = now
                trace(f"Typing dots [{chat}]: already on — kept on")
                return TypingResult(True)
            self._close(chat, window)       # ... and the chat's window is opened anew
        if not _plain(name):
            return self._gave_up(chat, "the chat has no name to find it by")
        if self._retry_after.get(chat, 0) > now:
            return self._gave_up(chat, f"its sidebar row opened another chat's window a moment ago — not tried "
                                       f"again for {RETRY_SECONDS / 60:.0f} min")
        if len(self._windows) >= MAX_WINDOWS:
            return self._refused(chat, TYPING_BUSY, f"{MAX_WINDOWS} chats show dots already")
        return self._open(chat, name, look, now)

    def _open(self, chat: str, name: str, look: _Look, now: float) -> TypingResult:
        """Find the chat's row, open its window, check it, write the field."""
        others = [w for w in look.windows if not self._ax.same(w, look.main) and not self._is_mine(w)]
        if any(same_name(self._ax.attribute(w, TITLE), name) for w in others):
            return self._gave_up(chat, f"'{name}' is already open in a window of its own that Keep did not open — "
                                       f"it is never typed into; close it for dots")
        try:
            rows = self._rows(look.sidebar, name)
        except _Capped as e:
            return self._gave_up(chat, f"{e} — not the sidebar of Messages as it is known")
        if len(rows) != 1:
            which = "no visible row" if not rows else f"{len(rows)} visible rows"
            return self._refused(chat, TYPING_NO_ROW, f"{which} of the Messages sidebar could be '{name}' and no "
                                                      f"other chat — a chat is found only among the rows the sidebar "
                                                      f"shows, by its exact name")
        row, action = rows[0]
        leading = self._ax.attribute(row, DESCRIPTION)
        if not self._ax.perform(row, action):
            return self._gave_up(chat, f"the row of '{name}' did not take '{OPEN_ACTION}'")
        opened = None
        try:
            opened, strays = self._await_window(name, look)
            if opened is None:
                return self._not_the_chats(chat, name, leading, strays, now)
            field = self._find(opened, COMPOSE_FIELD)
            if field is None:
                raise _NotAsExpected("it has no compose field")
            window = self._windows[chat] = _Window(name, opened, field, now)
            if not self._exit_hooked:
                self._exit_hooked = True
                atexit.register(self.shutdown)
            unexpected = self._unexpected(window)
            if unexpected:
                raise _NotAsExpected(unexpected)
            window.written = True
            if not self._ax.set_attribute(field, VALUE, VALUES[0]):
                raise _NotAsExpected("its compose field did not take a value")
        except BaseException as e:
            # Whatever went wrong after the press — an interrupt of the wait for the window
            # included — the window it opened does not stay behind.
            if chat in self._windows:
                self._close(chat, self._windows[chat])
            elif opened is not None:
                self._press_close(opened)
            else:
                self._close_what_the_press_opened(name, look)
            if not isinstance(e, Exception):
                raise
            why = str(e) if isinstance(e, _NotAsExpected) else f"{type(e).__name__}: {e}"
            return self._gave_up(chat, f"the window of '{name}' was not as expected ({why}) — closed again")
        self._schedule()
        if self._told.pop(chat, None):
            info(f"Typing dots [{chat}]: showing again, in the window '{name}'")
        else:
            trace(f"Typing dots [{chat}]: on, in the window '{name}'")
        return TypingResult(True)

    def _not_the_chats(self, chat: str, name: str, leading, strays: list, now: float) -> TypingResult:
        """No window titled with the chat's name came of the press. A window that did appear
        is closed only when it is provably what the press opened: its title is what the
        pressed row's own description leads with — the row was a longer-named chat's. Any
        other new window may be one the user opened in that moment and is left alone. Either
        way the row is not pressed again for a while."""
        ours = [w for w in strays if names_row(leading, _plain(self._ax.attribute(w, TITLE)))]
        if not strays:
            return self._gave_up(chat, f"no window titled '{name}' opened within {WINDOW_WAIT:.0f} s")
        self._retry_after[chat] = now + RETRY_SECONDS
        if len(ours) != 1:
            return self._gave_up(chat, f"no window titled '{name}' opened within {WINDOW_WAIT:.0f} s — "
                                       f"{len(strays)} other window(s) appeared and were left alone, since none is "
                                       f"provably the one its row opened")
        title = _plain(self._ax.attribute(ours[0], TITLE))
        self._press_close(ours[0])
        if len(self._longer) >= KNOWN_LONGER:
            self._longer.clear()
        self._longer.setdefault(_plain(name), set()).add(title)
        return self._gave_up(chat, f"the row that could be '{name}' is the chat '{title}' — its window was closed "
                                   f"again, nothing was typed, and that row is not taken for '{name}' again")

    def _close_what_the_press_opened(self, name: str, look: _Look):
        """The wait for the chat's window was cut short: one more look, and a new window
        titled with the chat's name — the press's own — is closed."""
        try:
            new = [w for w in self._app_windows(look.app) if not any(self._ax.same(w, old) for old in look.windows)]
            for window in [w for w in new if same_name(self._ax.attribute(w, TITLE), name)][:1]:
                self._press_close(window)
        except Exception as e:
            error(f"Typing dots: the window of '{name}' may be left open ({type(e).__name__}: {e})")

    def _await_window(self, name: str, look: _Look) -> tuple[Any, list]:
        """(the one new window titled with the chat's name, the other new windows) once it is
        there; (None, every new window) when the wait ends without exactly one. New is what
        was not there at the look before the press: a window the user had open is never it."""
        deadline = self._clock() + WINDOW_WAIT
        while True:
            new = [w for w in self._app_windows(look.app)
                   if not any(self._ax.same(w, old) for old in look.windows)]
            named = [w for w in new if same_name(self._ax.attribute(w, TITLE), name)]
            if len(named) == 1:
                return named[0], [w for w in new if w is not named[0]]
            if self._clock() >= deadline:
                return None, new
            self._sleep(WINDOW_POLL)

    def _unexpected(self, window: _Window) -> str:
        """Why the window can no longer be taken for the chat's, "" while it can. Checked
        before every write."""
        title = self._ax.attribute(window.element, TITLE)
        if title is None:
            return "its window is gone"
        if not same_name(title, window.name):
            return "its window is titled otherwise now"
        if self._ax.attribute(window.field, IDENTIFIER) != COMPOSE_FIELD:
            return "its compose field is gone"
        return ""

    # --- keeping alive, timing out, cleaning up ---------------------------------------------

    def _schedule(self):
        if self._windows and self._timer is None:
            self._timer = self._start_timer(KEEPALIVE_SECONDS, self._tick)

    def _tick(self):
        with self._lock:
            self._timer = None
            try:
                self._keep_alive()
            except Exception as e:
                # Said once: with nothing tracked no timer runs, and the next reply starts over.
                error(f"Typing dots: the keep-alive failed ({type(e).__name__}: {e}) — {len(self._windows)} "
                      f"window(s) of Messages may be left open with a draft")
                self._windows.clear()
            self._schedule()

    def _keep_alive(self):
        gate, detail, _ = self._session_gate()
        now = self._clock()
        for chat, window in list(self._windows.items()):
            if gate:
                self._wait_for_the_mac(chat, window, detail)
            elif not window.live:
                self._close(chat, window)
            elif now - window.since >= TIMEOUT_SECONDS:
                info(f"Typing dots [{chat}]: nobody turned them off within {TIMEOUT_SECONDS:.0f} s — taken away")
                self._close(chat, window)
            else:
                self._rewrite(chat, window)

    def _rewrite(self, chat: str, window: _Window):
        unexpected = self._unexpected(window)
        if not unexpected:
            window.flip ^= 1
            if self._ax.set_attribute(window.field, VALUE, VALUES[window.flip]):
                return
            unexpected = "its compose field did not take a value"
        self._close(chat, window)
        self._gave_up(chat, f"{unexpected} — the dots of '{window.name}' end here")

    def _wait_for_the_mac(self, chat: str, window: _Window, detail: str):
        """Messages cannot be reached (the screen locked meanwhile): the dots die by themselves
        and the window is cleared and closed by the first look that can."""
        if window.live:
            window.live = False
            info(f"Typing dots [{chat}]: cannot be kept or taken away — {detail}. They stop by themselves; the "
                 f"window '{window.name}' is closed once Messages can be reached.")
        self._schedule()

    def _sweep(self):
        for chat, window in list(self._windows.items()):
            if not window.live:
                self._close(chat, window)

    def _close(self, chat: str, window: _Window):
        """Clear the field and close the window — of whatever chat it shows by now: it is this
        object's window and what the field holds was written here."""
        self._windows.pop(chat, None)
        if not self._windows and self._timer is not None:
            self._timer.cancel()        # nothing left to keep alive or to close
            self._timer = None
        if self._ax.attribute(window.element, ROLE) is None:
            trace(f"Typing dots [{chat}]: the window '{window.name}' does not answer (closed by hand, or Messages "
                  f"cannot be reached) — nothing cleared")
            return
        cleared = not window.written or self._ax.set_attribute(window.field, VALUE, "")
        closed = self._press_close(window.element)
        if not (cleared and closed):
            error(f"Typing dots [{chat}]: the window '{window.name}' could not be "
                  f"{'cleared' if not cleared else 'closed'} — it may be left on the Mac's screen"
                  f"{' with a draft' if not cleared else ''}")

    def _press_close(self, window) -> bool:
        button = self._ax.attribute(window, CLOSE_BUTTON)
        return button is not None and self._ax.perform(button, PRESS)

    # --- saying why, once -------------------------------------------------------------------

    def _refused(self, chat: str, reason: str, detail: str) -> TypingResult:
        if self._told.get(chat) != detail:
            self._told[chat] = detail
            info(f"Typing dots [{chat}]: none — {detail}. The reply is sent all the same.")
        else:
            trace(f"Typing dots [{chat}]: still none — {detail}")
        return TypingResult(reason=reason, detail=detail)

    def _gave_up(self, chat: str, detail: str) -> TypingResult:
        return self._refused(chat, TYPING_FAILED, detail)
