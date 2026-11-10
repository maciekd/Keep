"""The interface to one Mac's Messages account, and the plain values that cross it.

Whoever reads or sends iMessages talks to a `MessagesHost`: `LocalMessagesHost` when Messages.app
runs on this Mac under this user, `RemoteMessagesHost` when it runs elsewhere, behind a satellite. So everything
here is made for a wire: the arguments and the results are dataclasses of str / int / float /
bool / None and lists of those (`Wire` turns them into JSON and back), a failure is a value in
the result — never an exception of sqlite or subprocess — and a file on the host's disk crosses
as a reference (`Attachment.id`) with a call that returns its bytes.

A host holds nothing about who is asking: the cursor (a ROWID), the horizon, the rows a caller
tracks and the caller's own name are arguments, so two consumers can use one host. What it does
keep describes the Mac: the contacts, the database's schema, the send rate of the account."""
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# HostHealth.reason
NO_DATABASE = "no_database"     # chat.db does not exist: Messages.app was never set up for this user
NO_ACCESS = "no_access"         # chat.db is there and cannot be read: macOS keeps the process out (Full Disk Access)
# ... and of a host on another Mac, before its chat.db is even asked about
NOT_CONFIGURED = "not_configured"   # no usable address or no token: nothing was sent
UNREACHABLE = "unreachable"     # no connection, or no answer in time
REFUSED = "refused"             # the satellite did not take the token
BAD_ANSWER = "bad_answer"       # an answer that cannot be used: another wire version, a failing satellite, not a satellite at all

# TypingResult.reason — why the typing dots do not show. The first six are gates: facts about the
# host's Mac, the same for every chat, also reported in HostHealth.typing.
TYPING_UNAVAILABLE = "unavailable"                  # the Accessibility frameworks cannot be loaded in this process
TYPING_NO_ACCESSIBILITY = "no_accessibility"        # macOS has not granted the process Accessibility
TYPING_NOT_ON_CONSOLE = "not_on_console"            # the user's session is not the one on the Mac's screen
TYPING_SCREEN_LOCKED = "screen_locked"              # a locked session exposes no window
TYPING_NO_MESSAGES = "messages_not_running"         # Messages.app is not open
TYPING_NO_WINDOW = "no_messages_window"             # Messages shows no window with its conversation list
TYPING_GATES = (TYPING_UNAVAILABLE, TYPING_NO_ACCESSIBILITY, TYPING_NOT_ON_CONSOLE, TYPING_SCREEN_LOCKED,
                TYPING_NO_MESSAGES, TYPING_NO_WINDOW)
# ... and the rest are about one chat, one time
TYPING_NO_NAME = "no_name"      # the host cannot tell what its Messages sidebar calls the chat, or the name is not one chat's alone
TYPING_NO_ROW = "no_row"        # not exactly one visible row of the sidebar carries the chat's name
TYPING_BUSY = "busy"            # as many chats show dots as the host keeps windows open for
TYPING_FAILED = "failed"        # anything else: a window that is not the chat's, a field that is gone, a call that failed

# Message.schedule_state
SCHEDULE_COMPLETED = "completed"
SCHEDULE_FAILED = "failed"


@dataclass
class SatelliteStatus:
    """What the process serving a host on another Mac says about itself."""
    wire: int = 0                   # the wire version it speaks
    version: str = ""               # its `Version.detail()`
    branch: str = ""                # the branch it started on, "" for a detached checkout
    commit: str = ""                # the commit it started on, "" when git could not say
    updating: bool = False          # it is bringing itself to its caller's version right now
    update_failure: str = ""        # why its last try at that left it where it was, "" when none did


@dataclass
class HostHealth:
    """Whether the host can read its Messages database, and how many contacts it knows."""
    ok: bool = False
    reason: str = ""                # one of the reasons above when not ok
    detail: str = ""                # the fact behind the reason, naming the file or the address — for the log and the user
    contacts: int = 0               # handles in the contact cache
    satellite: Optional[SatelliteStatus] = None     # set by a satellite that answered; None for a host in this process
    # The gate (one of TYPING_GATES) that kept the typing dots from showing when they were last
    # asked for and still does; "" when they can show, or when nobody has asked this host for dots.
    typing: str = ""


@dataclass
class Count:
    """A number read from chat.db — a ROWID or a row count — or why it could not be read."""
    value: int = 0
    error: str = ""


@dataclass
class Attachment:
    """A file attached to a message, as a reference: `MessagesHost.attachment` returns its bytes."""
    id: int                         # attachment.ROWID
    name: str = ""                  # the file's name without its folder
    mime_type: Optional[str] = None


@dataclass
class Message:
    """One row of chat.db's message table, decoded and with its people named."""
    rowid: int
    timestamp: float = 0.0          # unix seconds; for a Send Later row the time it is set to send
    is_from_me: bool = False
    text: Optional[str] = None      # the body, stripped; read from the attributed body when the text column is empty
    handle: Optional[str] = None    # the sender's phone or email; None on an outgoing row
    sender: str = ""                # the sender's contact name, the handle when no contact has it
    chat_name: Optional[str] = None     # the display name of a named group chat
    # the chat's other participants by contact name, sorted — on an outgoing row of a chat without a name
    recipients: list[str] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)     # the files that exist on the host's disk
    send_later: bool = False        # still marked as a Send Later message: composed, not sent
    schedule_state: str = ""        # SCHEDULE_COMPLETED / SCHEDULE_FAILED, "" otherwise
    in_chat: bool = True            # attached to a chat; a copy iCloud syncs down may not be
    # What a consumer that answers in the chat needs (the chat interface); filled by `poll` alone,
    # and left at these defaults by a host one version behind.
    guid: str = ""                  # message.guid
    chat_guid: str = ""             # chat.guid of the chat it is in — what `send_chat` addresses
    is_group: bool = False          # the chat is a group conversation (chat.style)
    service: str = ""               # message.service: "iMessage", "SMS", "RCS"
    # The bubble this one refers to: the one whose thread it replies in, or the one a tapback was put on.
    reply_to: str = ""              # that bubble's message.guid, "" for a message that refers to none
    reply_to_from_me: bool = False  # that bubble is one of the account's own
    reply_to_text: Optional[str] = None     # its body, when the row is still there
    tapback: int = 0                # message.associated_message_type: 2000–2999 a tapback added, 3000–3999 one removed
    # Files chat.db lists for the row that are not on the host's disk (yet): Messages writes the row
    # before a large attachment has finished downloading. Left at 0 by a host one version behind.
    pending_attachments: int = 0


@dataclass
class MessageBatch:
    """The rows one read returned, with where the reader stands after them."""
    messages: list[Message] = field(default_factory=list)
    cursor: int = 0                 # the highest ROWID the read passed; what the next poll starts after
    exhausted: bool = True          # False: the read stopped at its limit, more is waiting
    error: str = ""


@dataclass(frozen=True)
class GroupChat:
    """A group conversation of chat.db, named the way the feed names it."""
    guid: str = ""                  # chat.guid — also the chat's AppleScript id, what a group send addresses
    name: Optional[str] = None      # the display name, None for an unnamed group
    members: list[str] = field(default_factory=list)    # contact names, sorted — the "Alice, Bob" of an unnamed group
    last_active: int = 0            # Apple date of its latest message; the tiebreak between look-alike groups
    handles: list[str] = field(default_factory=list)    # the members' phones and emails as chat.db has them, sorted

    @property
    def label(self) -> str:
        return self.name or ", ".join(self.members)

    def describe(self) -> str:
        members = ", ".join(self.members)
        return f"{self.name} ({members})" if self.name else members


@dataclass
class Recipient:
    """What a recipient string resolved to: the handle a send addresses, or why there is none."""
    handle: Optional[str] = None    # an E.164 phone, a lowercase email, or a group chat's guid
    error: str = ""
    name: Optional[str] = None      # the contact name the feed shows for the handle, None when no contact has it
    full_name: Optional[str] = None     # the card's "First Middle Last"
    group: Optional[GroupChat] = None   # set when the handle is a group chat's guid
    no_contacts: bool = False       # the lookup failed because the host has no contacts at all (AddressBook unread)


@dataclass
class SendResult:
    """How a send went: `error` is empty when Messages took the message."""
    error: str = ""
    recipient: Optional[Recipient] = None   # who it was resolved to, once the resolution was reached
    service: str = ""               # what Messages reported: "sent:imessage", "sent:sms", "sent:group" or "sent"
    unauthorized: bool = False      # macOS refused the Apple event: the Automation permission is missing


@dataclass
class GroupList:
    """The group chats of the account, or why they could not be read."""
    groups: list[GroupChat] = field(default_factory=list)
    error: str = ""


@dataclass
class TypingResult:
    """Whether the typing dots show in a chat after a `typing` call. Dots that do not show are
    never a failure of anything else: the reply is sent all the same."""
    showing: bool = False
    reason: str = ""                # one of the TYPING_* reasons when they were asked for and do not show
    detail: str = ""                # the fact behind the reason, in words for the log
    error: str = ""                 # the call itself did not go through (a satellite out of reach, a host that failed)


@dataclass
class AttachmentData:
    """The bytes of an attachment, base64-encoded for the wire."""
    data: str = ""
    error: str = ""


class MessagesHost(ABC):
    """One Mac's Messages account: its chat.db to read, its contacts to resolve, its Messages.app to send through."""

    # Where the account's Mac is reached when it is not this process's own: a consumer words
    # its hints by it (whose Mac needs a grant). "" for a host in this process.
    address: str = ""

    @abstractmethod
    def health(self, refresh: bool = False) -> HostHealth:
        """Whether chat.db can be read. With `refresh` a healthy host also re-reads what it
        caches about the Mac (the contacts, the schema) — what a consumer asks for when it opens."""

    @abstractmethod
    def cursor_at(self, horizon_ts: Optional[float]) -> Count:
        """The ROWID just before the first message dated at or after the horizon; past every
        row when there is none."""

    @abstractmethod
    def last_rowid(self) -> Count:
        """The highest ROWID the message table ever gave out: the end of the table, whatever its
        rows are dated. A cursor past it is not a place in this database."""

    @abstractmethod
    def remaining(self, after_rowid: int, horizon_ts: Optional[float]) -> Count:
        """How many messages `poll` still has to read after the ROWID."""

    @abstractmethod
    def poll(self, after_rowid: int, horizon_ts: Optional[float], limit: int) -> MessageBatch:
        """The messages after the ROWID dated at or after the horizon (None: every row), in
        ROWID order, at most `limit` rows read. A row with neither text nor a file is left out;
        the batch's cursor has passed it all the same."""

    @abstractmethod
    def messages(self, rowids: list[int]) -> MessageBatch:
        """The given rows as they are now, without their attachments; a deleted row is missing
        from the answer. For a consumer that follows Send Later messages until they send."""

    @abstractmethod
    def scheduled(self) -> MessageBatch:
        """Every row still marked as a Send Later message — time, direction and state only."""

    @abstractmethod
    def attachment(self, attachment_id: int) -> AttachmentData:
        """The bytes of an attachment a `Message` referred to."""

    @abstractmethod
    def resolve(self, recipient: str, owner: str = "") -> Recipient:
        """The handle behind a contact name, a raw phone or email, a group's name or a list of a
        group's members. `owner` is the account holder's name as the caller knows it, ignored
        in a member list."""

    @abstractmethod
    def send(self, recipient: str, body: str, owner: str = "") -> SendResult:
        """Resolve the recipient and send the text through Messages.app."""

    @abstractmethod
    def send_chat(self, chat: str, body: str) -> SendResult:
        """Send the text to a chat as it is addressed, resolving nothing: a `Message.chat_guid`
        — the exact group, or the 1:1 chat — or a buddy's phone or email. For a consumer that
        answers where it was written to; its sends have a rate limit of their own."""

    @abstractmethod
    def groups(self) -> GroupList:
        """Every group chat of the account, with its members' names and handles."""

    @abstractmethod
    def typing(self, chat: str, on: bool) -> TypingResult:
        """Show the typing dots to the people of a chat (`Message.chat_guid`) while a reply is
        worked on, or take them away. The host keeps them alive by itself and takes them away
        by itself when no `off` comes within its safety timeout, so a turn costs one `on` and
        one `off`. Best effort by contract: the dots show only when the host is sure which
        chat it types in, and every reason they do not is a value."""

    def close(self) -> None:
        """Take back what the host put on its Mac's screen — the dots, their windows — when the
        process that holds it ends. Nothing for a host that is elsewhere."""

    def away(self) -> bool:
        """Whether the caller's last call — the calling thread's own — found the host out of
        reach altogether (no connection, no answer, a satellite on a wire version it is about to
        leave) rather than failing at something. A consumer that polls keeps its place and asks
        again instead of counting a failure. Never so for a host in this process."""
        return False

    def attachment_size(self, attachment_id: int) -> Count:
        """How many bytes an attachment's file has, for a caller with a limit to ask before it
        asks for the bytes — a file over the limit is then never read. A host that cannot tell
        without reading answers with an error, and the caller measures what it was handed."""
        return Count(error="this host does not tell an attachment's size")

    def read_attachment(self, attachment_id: int) -> tuple[Optional[bytes], str]:
        """(bytes, "") or (None, why not) — `attachment` decoded; a host with the file at hand
        skips the encoding."""
        answer = self.attachment(attachment_id)
        if answer.error:
            return None, answer.error
        return base64.b64decode(answer.data), ""
