"""What the iMessage chat's tests share: the cast and its handles, the config lists, rows as a
host reads them, and a hub that only records."""
from contextlib import contextmanager
from unittest.mock import patch

from common.ConfigManager import config
from common.imessage.MessagesHost import Message

NOW = 1_794_300_000.0
MARCUS, CHARLOTTE, ALICE = "+46701740605", "+46701740606", "alice@example.com"
BOBS_KEEP, STRANGER = "+46701740610", "+46701740699"
DIRECT = f"iMessage;-;{MARCUS}"             # Marcus's 1:1 chat with the assistant
CHARLOTTES = f"iMessage;-;{CHARLOTTE}"
FAMILY, CHESS = "iMessage;+;chat-family", "iMessage;+;chat-chess"

LISTS = {
    "imessage_chat.owners": [f"Marcus {MARCUS}"],
    "imessage_chat.guests": [f"Charlotte {CHARLOTTE}", f"Alice {ALICE}"],
    "imessage_chat.bots": [f"Bob's Keep {BOBS_KEEP}"],
}


@contextmanager
def configured(owner_feed: bool = False, **values):
    """config.yaml with the cast on the lists; `values` (written `section__key`) on top. The
    iMessage integration — the other consumer of a host — is off unless `owner_feed`."""
    keys = {**LISTS, **{name.replace("__", "."): value for name, value in values.items()}}
    real = config.get

    def get(key, default=None):
        if key in keys:
            return keys[key]
        return default if key.split(".")[0] in ("imessage", "imessage_chat") else real(key, default)

    with patch.object(config, "get", side_effect=get), \
            patch.object(config, "integration_enabled", side_effect=lambda name, default=False: owner_feed):
        yield keys


class Clock:
    def __init__(self, now: float = NOW):
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeHub:
    """Takes what the transport hands over, or refuses it as a session does: with `taken` off
    without a reason (a guest's debug command), with `busy` on because a reply is in flight —
    the transport is told (`turned_away`) before the refusal returns."""

    def __init__(self):
        self.received = []          # everything handed over, taken or not
        self.accepted = []          # what became a turn
        self.recorded = []
        self.taken = True
        self.busy = False

    def receive(self, transport, message) -> bool:
        self.received.append(message)
        if self.busy and message.guest:
            transport.turned_away(message.conversation, [message.message_id])
            return False
        if self.taken:
            self.accepted.append(message)
        return self.taken

    def record(self, transport, conversation, direction, text):
        self.recorded.append((conversation, direction, text))

    def unregister(self, transport):
        pass

    @property
    def texts(self) -> list[str]:
        return [message.text for message in self.received]


def text(rowid, body, handle=MARCUS, chat=None, ago=5.0, group=None, service="iMessage", **fields) -> Message:
    """An incoming row: in the sender's 1:1 chat, or in `chat` — a group when its guid says so."""
    chat = chat or f"iMessage;-;{handle}"
    is_group = ";+;" in chat if group is None else group
    return Message(rowid, NOW - ago, False, body, handle, sender=handle,
                   chat_name="Family" if chat == FAMILY else None, guid=f"G-{rowid}", chat_guid=chat,
                   is_group=is_group, service=service, **fields)


def own(rowid, body, chat=FAMILY, ago=5.0) -> Message:
    """One of the assistant's own bubbles, as the poll reads it back."""
    return Message(rowid, NOW - ago, True, body, guid=f"OWN-{rowid}", chat_guid=chat, is_group=";+;" in chat,
                   service="iMessage", chat_name="Family" if chat == FAMILY else None)
