#!/usr/bin/env python3
"""The live check of Keep's iMessage host: the facts only an unlocked Mac with a phone beside it can show.

Run by a person — never by the server, never by a test — in Terminal, at the Mac and as the macOS
user whose Messages is the account to check, with the venv active:

    python tools/imessage_live_check/imessage_live_check.py +46701740605

The argument is a handle of your own, a phone number with + and the country code or the email
your iMessage uses: the check shows typing dots to that handle and to nobody else, so you watch
them arrive on your own phone. Without a handle it does nothing at all.

  1. The gates of the typing dots, each by itself: Accessibility granted to this Terminal, this
     user's session on the Mac's screen, the screen unlocked, Messages open with its main window.
  2. chat.db, read only: whether every column Keep reads is there, and counts — of messages,
     chats by kind, services, replies, tapbacks. Never a message's text, never a handle.
  3. The dots, through the code the server uses (`LocalMessagesHost.typing`): on for 20 seconds
     in the chat with your handle, then off. You say what the phone showed.
  4. Only with --send-test-message: one text to that same handle after the dots are gone, to
     see that the bubble arrives with no dots after it and no draft left in Messages.

--report-only stops after step 2 and touches nothing of Messages' windows; it sends nothing, so
it is refused together with --send-test-message. The exit code is 0 when every step that ran
went as expected, 1 when one did not, 2 when the check refused to run."""
import argparse
import os
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DOTS_SECONDS = 20
TEST_MESSAGE = "Keep live check: a test message. Nothing to do."
# The columns Keep's host layer reads (common/imessage/MessagesDatabase.py), by table.
COLUMNS = {
    "message": ["ROWID", "guid", "text", "attributedBody", "date", "is_from_me", "handle_id", "is_system_message",
                "service", "thread_originator_guid", "associated_message_guid", "associated_message_type"],
    "handle": ["ROWID", "id", "service"],
    "chat": ["ROWID", "guid", "style", "display_name"],
    "chat_message_join": ["chat_id", "message_id"],
    "chat_handle_join": ["chat_id", "handle_id"],
    "attachment": ["ROWID", "filename", "mime_type"],
    "message_attachment_join": ["message_id", "attachment_id"],
}
# Not on every macOS: without them Send Later messages are not followed, nothing else changes.
OPTIONAL_COLUMNS = {"message": ["schedule_type", "schedule_state"]}
# Counts that show whether the columns hold what Keep assumes — numbers and service names only.
COUNTS = [
    ("messages", "SELECT COUNT(*) FROM message"),
    ("1:1 chats (chat.style 45)", "SELECT COUNT(*) FROM chat WHERE style = 45"),
    ("group chats (chat.style 43)", "SELECT COUNT(*) FROM chat WHERE style = 43"),
    ("chats of any other style (expected 0)", "SELECT COUNT(*) FROM chat WHERE style NOT IN (43, 45)"),
    ("1:1 chats whose guid has ';-;'", "SELECT COUNT(*) FROM chat WHERE style = 45 AND guid LIKE '%;-;%'"),
    ("group chats whose guid has ';+;'", "SELECT COUNT(*) FROM chat WHERE style = 43 AND guid LIKE '%;+;%'"),
    ("named group chats", "SELECT COUNT(*) FROM chat WHERE style = 43 AND COALESCE(display_name, '') != ''"),
    ("messages without a guid (expected 0)", "SELECT COUNT(*) FROM message WHERE COALESCE(guid, '') = ''"),
    ("replies in a thread (thread_originator_guid set)",
     "SELECT COUNT(*) FROM message WHERE COALESCE(thread_originator_guid, '') != ''"),
    ("... whose thread_originator_guid is the guid of a message that is there",
     "SELECT COUNT(*) FROM message r WHERE COALESCE(r.thread_originator_guid, '') != '' "
     "AND EXISTS (SELECT 1 FROM message o WHERE o.guid = r.thread_originator_guid)"),
    ("tapbacks added (associated_message_type 2000-2999)",
     "SELECT COUNT(*) FROM message WHERE associated_message_type BETWEEN 2000 AND 2999"),
    ("tapbacks removed (3000-3999)", "SELECT COUNT(*) FROM message WHERE associated_message_type BETWEEN 3000 AND 3999"),
    ("... tapbacks whose associated_message_guid starts 'p:<n>/' or 'bp:'",
     "SELECT COUNT(*) FROM message WHERE associated_message_type BETWEEN 2000 AND 3999 "
     "AND (associated_message_guid LIKE 'p:%/%' OR associated_message_guid LIKE 'bp:%')"),
]
GROUPED = [
    ("messages by service", "SELECT COALESCE(service, '(none)'), COUNT(*) FROM message GROUP BY 1 ORDER BY 2 DESC"),
    ("chats by the first part of their guid",
     "SELECT substr(guid, 1, instr(guid, ';') - 1), COUNT(*) FROM chat WHERE instr(guid, ';') > 0 GROUP BY 1 ORDER BY 2 DESC"),
]
_PHONE = re.compile(r"^\+\d{8,15}$")
_EMAIL = re.compile(r"^[^@\s;,]+@[^@\s;,]+\.[^@\s;,]+$")


def own_handle(text: str) -> str:
    """The handle as Messages addresses it — a phone with + and digits only, an email in lower
    case — or "" for anything that is not one person's handle: a name, a number without its
    country code, a group chat's guid."""
    text = (text or "").strip()
    if _EMAIL.match(text):
        return text.lower()
    phone = re.sub(r"[ ()\-.]", "", text)
    return phone if _PHONE.match(phone) else ""


def parse(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="imessage_live_check.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("handle", nargs="?", default="",
                        help="a handle of your own: +<country code><number> or your iMessage email")
    parser.add_argument("--send-test-message", action="store_true",
                        help="after the dots, send one test text to that same handle")
    parser.add_argument("--report-only", action="store_true",
                        help="steps 1 and 2 only: no window is opened, no dots are shown")
    parser.add_argument("--seconds", type=int, default=DOTS_SECONDS, help="how long the dots stay on (default 20)")
    return parser.parse_args(argv)


def report_gates(ax, indicator, say) -> bool:
    """Step 1. Each gate by itself, so one run names everything that is missing."""
    from common.imessage.TypingIndicator import MESSAGES_BUNDLE
    say("\n[1] The gates of the typing dots")
    try:
        trusted = ax.trusted()
    except ImportError as e:
        say(f"    FAIL  the Accessibility frameworks (pyobjc) cannot be loaded: {e}")
        return False
    say(f"    {'ok  ' if trusted else 'FAIL'}  Accessibility is {'granted' if trusted else 'NOT granted'} to the app "
        f"this runs under" + ("" if trusted else " — System Settings > Privacy & Security > Accessibility, turn on "
                              "this terminal app"))
    session = ax.session()
    on_console = session is not None and session.on_console
    say(f"    {'ok  ' if on_console else 'FAIL'}  this user's session is {'' if on_console else 'NOT '}the one on the "
        f"Mac's screen")
    unlocked = session is not None and not session.locked
    say(f"    {'ok  ' if unlocked else 'FAIL'}  the screen is {'unlocked' if unlocked else 'LOCKED'}")
    running = ax.application(MESSAGES_BUNDLE) is not None
    say(f"    {'ok  ' if running else 'FAIL'}  Messages is {'running' if running else 'NOT running'}")
    if not (trusted and on_console and unlocked and running):
        return False
    rows, why = indicator.sidebar()
    if why:
        say(f"    FAIL  the sidebar of Messages cannot be read: {why}")
        return False
    say(f"    ok    the main window's sidebar is readable: {len(rows)} rows visible (only these can get dots)")
    return True


def report_database(path: str, say) -> bool:
    """Step 2. Columns by name, rows by count; nothing a person wrote or is called."""
    say(f"\n[2] chat.db, read only ({path})")
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.execute("SELECT 1 FROM message LIMIT 1").fetchone()
    except sqlite3.Error as e:
        say(f"    FAIL  it cannot be read ({e}) — Full Disk Access for this terminal app, or Messages was never "
            f"set up for this user")
        return False
    complete = True
    try:
        for table, wanted in COLUMNS.items():
            there = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            missing = [column for column in wanted if column not in there]
            complete = complete and not missing
            say(f"    {'ok  ' if not missing else 'FAIL'}  {table}: "
                + (f"all {len(wanted)} columns Keep reads" if not missing else f"MISSING {', '.join(missing)}"))
            optional = [column for column in OPTIONAL_COLUMNS.get(table, []) if column not in there]
            if optional:
                say(f"    note  {table}: no {', '.join(optional)} — Send Later messages are not followed on this macOS")
        if not complete:
            return False
        for label, sql in COUNTS:
            say(f"    {conn.execute(sql).fetchone()[0]:>8}  {label}")
        try:
            counter = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'message'").fetchone()
        except sqlite3.Error:
            counter = None
        say(f"    {counter[0]:>8}  the message table's ROWID counter (sqlite_sequence)" if counter else
            "    note  the message table keeps no ROWID counter: after the newest message is deleted, a restart "
            "reads the feed's horizon again — tell the developers")
        for label, sql in GROUPED:
            say(f"    {label}: " + (", ".join(f"{name or '(empty)'} {count}" for name, count in conn.execute(sql)) or "none"))
    finally:
        conn.close()
    return True


def _words(name: str) -> frozenset[str]:
    """The words of a name as two spellings of it are compared: composed, without direction
    marks, whatever the case and the order."""
    composed = unicodedata.normalize("NFC", name or "")
    return frozenset("".join(char for char in composed if unicodedata.category(char) != "Cf").casefold().split())


def rows_spelling(handle: str, rows: list[str], expected: str = "") -> list[str]:
    """How the sidebar names the rows that are this handle's own — the start of a row up to its
    first comma, and only of a row that carries there the handle itself or, with a contact card,
    the words of `expected` (the name this Mac expects the row to have) in any order and case:
    no other chat's name or message is ever shown."""
    digits = re.sub(r"\D", "", handle)
    wanted = _words(expected)

    def own(name: str) -> bool:
        if wanted and _words(name) == wanted:
            return True
        if "@" in handle:
            return handle in name.lower()
        return len(digits) >= 8 and digits[-8:] in re.sub(r"\D", "", name)
    return [name for name in (row.split(",")[0].strip() for row in rows) if own(name)]


def messages_windows(ax) -> list:
    """The windows Messages has right now; none when it does not run or does not answer."""
    from common.imessage.TypingIndicator import MESSAGES_BUNDLE, WINDOWS
    app = ax.application(MESSAGES_BUNDLE)
    windows = ax.attribute(app, WINDOWS) if app is not None else None
    return list(windows) if isinstance(windows, list) else []


def close_window_left_opening(ax, before: list, name: str, say, sleep) -> None:
    """After a Ctrl-C while the dots were being put on. Between the press on the chat's sidebar
    row and the moment the typing code takes the window that opens for its own, nothing knows
    that window, and neither `typing(chat, False)` nor the host's `close()` closes it. So it is
    closed here: a window of Messages that was not there before the dots were asked for and is
    titled with the chat's name, waited for as long as a window may take to open. Nothing was
    typed in it — a window the typing code had taken was cleared and closed by it before this."""
    from common.imessage.TypingIndicator import CLOSE_BUTTON, PRESS, TITLE, WINDOW_POLL, WINDOW_WAIT, same_name
    if not name:
        return
    try:
        left = []
        for _ in range(int(WINDOW_WAIT / WINDOW_POLL) + 1):
            left = [window for window in messages_windows(ax)
                    if not any(ax.same(window, old) for old in before) and same_name(ax.attribute(window, TITLE), name)]
            if left:
                break
            sleep(WINDOW_POLL)
        for window in left:
            button = ax.attribute(window, CLOSE_BUTTON)
            closed = button is not None and ax.perform(button, PRESS)
            say(f"    Interrupted while the window of {name!r} was opening: " + (
                "closed again, nothing was typed in it." if closed else
                "it could NOT be closed — close that small window by hand."))
        if not left:
            say("    Interrupted before a window of the chat was open: nothing to close.")
    except Exception as e:
        say(f"    Interrupted, and Messages could not be looked at for a window left open ({type(e).__name__}: {e}) "
            f"— if a small window titled {name!r} is on the screen, close it by hand.")


def confirmed(ask, question: str) -> bool:
    return ask(f"    {question} [y/n] ").strip().lower().startswith("y")


def show_dots(host, ax, chat: str, handle: str, seconds: int, say, ask, sleep) -> bool:
    """Step 3. `on`, the keep-alive over the wait, `off` — the calls a reply makes."""
    say(f"\n[3] Typing dots to {handle} for {seconds} s — have that phone open on the chat with this account")
    name, why = host.contacts.sidebar_name(chat)
    say(f"    This Mac expects the chat's sidebar row and window to be named: {name!r}" if name
        else f"    This Mac cannot tell what its sidebar calls that chat: {why}")
    before = messages_windows(ax)
    opening = True
    try:
        result = host.typing(chat, True)
        opening = False
        if not result.showing:
            say(f"    FAIL  no dots ({result.reason}): {result.detail or result.error}")
            spelled = rows_spelling(handle, host.typing_indicator.sidebar()[0], name)
            if spelled:
                say(f"          the sidebar spells the row of this chat {spelled} — tell the developers")
            return False
        say("    The dots are on. Watch the phone...")
        for left in range(seconds, 0, -5):
            say(f"      {left} s left")
            sleep(min(5, left))
    finally:
        host.typing(chat, False)
        if opening:     # only a Ctrl-C ends `typing` without an answer
            close_window_left_opening(ax, before, name, say, sleep)
    say("    The dots are off: the field is cleared and the window closed.")
    held = confirmed(ask, f"Did the phone show the dots for the whole {seconds} s?")
    gone = confirmed(ask, "Did they disappear within a few seconds of the end?")
    tidy = confirmed(ask, "Is the small window gone from this Mac, and Messages itself not brought to the front?")
    say(f"    {'ok  ' if held and gone and tidy else 'FAIL'}  dots held: {held}, dots gone: {gone}, window gone: {tidy}")
    return held and gone and tidy


def send_test(host, chat: str, handle: str, say, ask) -> bool:
    """Step 4. A send right after the dots: the bubble, no dots after it, no draft."""
    say(f"\n[4] One test message to {handle}")
    result = host.send_chat(chat, TEST_MESSAGE)
    if result.error:
        say(f"    FAIL  not sent: {result.error}")
        return False
    say(f"    Sent ({result.service}).")
    clean = confirmed(ask, "Did the bubble arrive with no dots after it, and is the chat's compose field in Messages empty?")
    say(f"    {'ok  ' if clean else 'FAIL'}  clean after the send: {clean}")
    return clean


def main(argv=None, host=None, ax=None, say=print, ask=input, sleep=time.sleep) -> int:
    args = parse(argv)
    handle = own_handle(args.handle)
    if not handle:
        say("Refused: give a handle of your own — a phone number with + and the country code (+46701740605) or "
            "your iMessage email. The typing dots are shown to that handle and to nobody else; nothing is done "
            "without one." if not args.handle.strip() else
            f"Refused: {args.handle!r} is not one person's handle. A phone number with + and the country code, or "
            f"an email — never a name, a group or a number without its country code.")
        return 2
    if not 5 <= args.seconds <= 120:
        say("Refused: --seconds is between 5 and 120.")
        return 2
    if args.report_only and args.send_test_message:
        say("Refused: --report-only stops before anything is shown or sent, so --send-test-message cannot go "
            "with it. Leave one of the two out.")
        return 2
    os.chdir(ROOT)
    from common.imessage.Accessibility import Accessibility
    from common.imessage.LocalMessagesHost import LocalMessagesHost
    host = host if host is not None else LocalMessagesHost()
    ax = ax if ax is not None else Accessibility()
    chat = f"iMessage;-;{handle}"
    say(f"Keep iMessage live check — handle {handle}"
        + (", with a test message" if args.send_test_message else ", no message is sent"))
    gates = report_gates(ax, host.typing_indicator, say)
    database = report_database(host.database.path, say)
    if args.report_only:
        say("\n--report-only: no dots shown, nothing sent.")
        return 0 if gates and database else 1
    if not gates:
        say("\n[3] Skipped: the dots cannot show while a gate above fails." + (
            " [4] Skipped with it: the test message is for seeing the dots end." if args.send_test_message else ""))
        return 1
    try:
        dots = show_dots(host, ax, chat, handle, args.seconds, say, ask, sleep)
        sent = send_test(host, chat, handle, say, ask) if args.send_test_message and dots else True
        if args.send_test_message and not dots:
            say("\n[4] Skipped: the dots did not go as expected, so there is nothing for the message to show.")
    finally:
        host.close()
    return 0 if database and dots and sent else 1


if __name__ == "__main__":
    sys.exit(main())
