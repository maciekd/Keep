"""LocalMessagesHost and the MessagesHost contract it stands for: failures cross as values,
everything that crosses survives JSON, a file crosses as a reference, and the host keeps
nothing about who is asking."""
import ast
import base64
import dataclasses
import inspect
import json
from typing import Union, get_args, get_origin, get_type_hints
from unittest.mock import Mock, patch

import pytest

from common.Paths import SRC_DIR
from common.imessage import MessagesHost as contract
from common.imessage import MessagesSender as sender_module
from common.imessage.Contacts import Contacts
from common.imessage.LocalMessagesHost import LocalMessagesHost
from common.imessage.MessagesDatabase import MessagesDatabase
from common.imessage.MessagesHost import (Attachment, AttachmentData, Count, GroupChat, GroupList, HostHealth, Message,
                                          MessageBatch, MessagesHost, NO_DATABASE, Recipient, SatelliteStatus,
                                          SendResult, TYPING_FAILED, TYPING_NO_ACCESSIBILITY, TYPING_NO_NAME,
                                          TYPING_NO_ROW, TypingResult, UNREACHABLE)
from common.imessage.TypingIndicator import TypingIndicator
from common.imessage.Wire import from_wire, to_wire

from .ax_fakes import Clock, FakeMessages, Timers
from .messages_fakes import ChatDb, GROUP, NOW, address_book

PEOPLE = [("Alice", "Smith", [("070-174 06 05", "_$!<Mobile>!$_")], []),
          ("Bob", "Jones", [("070-174 06 06", "_$!<Mobile>!$_")], [])]


@pytest.fixture
def chat(tmp_path):
    db = ChatDb(tmp_path / "chat.db")
    db.handle(1, "+46701740605").handle(2, "+46701740606")
    db.chat(1, "any;-;+46701740605", [1]).chat(2, "any;+;chat2", [1, 2], name="Family", style=GROUP)
    photo = tmp_path / "IMG_0001.jpg"
    photo.write_bytes(b"\xff\xd8\xff fake jpeg \x00\x01")
    db.message(1, "Hello", ts=NOW - 300, chat=1, sender=1)
    db.message(2, "Look", ts=NOW - 200, chat=2, sender=2).attachment(40, 2, str(photo), "image/jpeg")
    db.message(3, "On my way", ts=NOW - 100, chat=1)
    db.message(4, "See you tomorrow", ts=NOW + 3600, chat=1, schedule_type=2)
    return db


@pytest.fixture
def host(chat, tmp_path):
    database = MessagesDatabase(chat.path)
    return LocalMessagesHost(database, Contacts(database, address_book(tmp_path, PEOPLE)))


def osascript(**result):
    answer = Mock(**{"returncode": 0, "stdout": "sent:imessage", "stderr": "", **result})
    return patch.object(sender_module.subprocess, "run", return_value=answer)


@pytest.mark.unit
class TestTheCalls:
    def test_health_reports_the_database_and_reads_the_contacts_only_when_asked(self, host):
        assert host.health() == HostHealth(True, contacts=0)
        opened = host.health(refresh=True)
        assert opened.ok and opened.contacts == 4, "two numbers, each with its 9-digit suffix"
        assert host.health().contacts == 4

    def test_health_names_the_accounts_own_addresses(self, host, chat):
        chat.execute("UPDATE chat SET last_addressed_handle = 'keep@example.com', account_login = 'E:keep@example.com'")
        assert host.health().addresses == ["keep@example.com"]

    def test_a_poll_names_the_people_from_the_address_book(self, host):
        batch = host.poll(0, None, 500)
        assert batch.error == "" and batch.exhausted and batch.cursor == 4
        hello, look, sent, later = batch.messages
        assert (hello.sender, look.sender, look.chat_name) == ("Alice", "Bob", "Family")
        assert sent.recipients == ["Alice"]
        assert later.send_later

    def test_a_poll_that_stops_at_its_limit_says_more_is_waiting(self, host):
        batch = host.poll(0, None, 2)
        assert not batch.exhausted and batch.cursor == 2
        assert host.poll(batch.cursor, None, 500).exhausted

    def test_the_start_and_the_count_ahead_follow_the_horizon(self, host):
        assert host.cursor_at(NOW - 150) == Count(2)
        assert host.last_rowid() == Count(max(m.rowid for m in host.poll(0, None, 500).messages))
        assert host.remaining(0, NOW - 250) == Count(3)

    def test_the_scheduled_rows_and_a_row_read_again(self, host):
        assert [m.rowid for m in host.scheduled().messages] == [4]
        (row,) = host.messages([4]).messages
        assert row.send_later and row.text == "See you tomorrow" and row.recipients == ["Alice"]

    def test_a_name_resolves_and_a_send_goes_to_its_handle(self, host):
        assert host.resolve("alice smith").handle == "0701740605"
        with osascript() as run:
            result = host.send("Alice", "hi")
        assert result.error == "" and result.service == "sent:imessage"
        assert result.recipient.handle == "0701740605" and result.recipient.full_name == "Alice Smith"
        assert '"0701740605"' in run.call_args[0][0][2]

    def test_a_group_resolves_by_name_and_is_sent_to_by_guid(self, host):
        with osascript(stdout="sent:group") as run:
            result = host.send("family", "dinner at 7")
        assert result.recipient.group == GroupChat("any;+;chat2", "Family", ["Alice", "Bob"],
                                                   result.recipient.group.last_active,
                                                   ["+46701740605", "+46701740606"])
        assert 'chat id "any;+;chat2"' in run.call_args[0][0][2]

    def test_an_unresolved_recipient_never_reaches_messages(self, host):
        with osascript() as run:
            result = host.send("Zelda", "hi")
        run.assert_not_called()
        assert "unknown contact" in result.error and result.recipient.handle is None

    def test_an_empty_body_is_refused_before_anything_is_looked_up(self, host):
        host.contacts = Mock()
        assert host.send("Alice", "  ").error == "empty message body"
        host.contacts.resolve.assert_not_called()

    def test_a_chat_send_addresses_the_group_by_the_guid_it_was_given(self, host):
        with osascript(stdout="sent:group") as run:
            result = host.send_chat("any;+;chat2", "dinner at 7")
        assert result.error == "" and result.service == "sent:group"
        assert 'chat id "any;+;chat2"' in run.call_args[0][0][2]

    def test_a_chat_send_to_a_direct_chat_goes_to_the_buddy_its_guid_ends_in(self, host):
        """The send that falls back to SMS takes a buddy, not a chat id."""
        with osascript() as run:
            assert host.send_chat("any;-;+46701740605", "hi").error == ""
        script = run.call_args[0][0][2]
        assert 'buddy "+46701740605"' in script and "any;-;" not in script

    def test_a_chat_send_resolves_nothing(self, host):
        """A guid looks like a phone number to the resolver, which would strip it to its digits."""
        host.contacts = Mock()
        with osascript():
            host.send_chat("any;+;chat2", "hi")
            host.send_chat("m@mlab.test", "hi")
        host.contacts.resolve.assert_not_called()

    def test_chat_sends_have_a_budget_of_their_own(self, host):
        """A conversation must not be stopped by the five sends a minute of send_imessage, nor use them up."""
        with osascript():
            for _ in range(sender_module.SEND_RATE_LIMIT_MAX + 3):
                assert host.send_chat("any;-;+46701740605", "hi").error == ""
            assert host.send("Alice", "hi").error == "", "the chat's bubbles counted against the account's sends"

    def test_the_chat_budget_ends_too(self, host):
        with osascript(), patch.object(sender_module, "CHAT_RATE_LIMIT_MAX", 2):
            host.sender = sender_module.MessagesSender()
            results = [host.send_chat("any;-;+46701740605", "hi").error for _ in range(3)]
        assert results[:2] == ["", ""] and "rate limit" in results[2]

    def test_the_groups_are_listed_with_the_handles_of_their_members(self, host):
        listing = host.groups()
        assert listing.error == ""
        assert [(g.guid, g.name, g.handles) for g in listing.groups] == \
            [("any;+;chat2", "Family", ["+46701740605", "+46701740606"])]

    def test_the_rate_limit_is_checked_before_the_recipient(self, host):
        host.sender.refusal = Mock(return_value="rate limit: 5 sends per 60s")
        host.contacts = Mock()
        assert "rate limit" in host.send("Alice", "hi").error
        host.contacts.resolve.assert_not_called()


@pytest.fixture
def mac():
    """Messages on the host's screen: the family group and Alice among the sidebar's rows."""
    messages = FakeMessages()
    messages.chat("Family")
    messages.chat("Alice Smith")
    return messages


@pytest.fixture
def typing_host(chat, tmp_path, mac):
    clock = Clock()
    database = MessagesDatabase(chat.path)
    return LocalMessagesHost(database, Contacts(database, address_book(tmp_path, PEOPLE)),
                             typing_indicator=TypingIndicator(mac, clock, clock.sleep, Timers(clock)))


@pytest.mark.unit
class TestTheTypingDots:
    """The caller names a chat by its guid; what the sidebar of the host's Mac calls it is the host's to know."""

    def test_a_group_is_found_by_its_name(self, typing_host, mac):
        assert typing_host.typing("any;+;chat2", True) == TypingResult(True)
        assert mac.opened == ["Family"]

    def test_a_person_is_found_by_the_full_name_on_the_hosts_own_contact_card(self, typing_host, mac):
        """The chat's guid carries the number; the card files it in another format."""
        assert typing_host.typing("any;-;+46701740605", True).showing
        assert mac.opened == ["Alice Smith"]

    def test_off_takes_them_away_again(self, typing_host, mac):
        typing_host.typing("any;+;chat2", True)
        assert typing_host.typing("any;+;chat2", False) == TypingResult()
        assert mac.closed == ["Family"] and mac.windows == [mac.main]

    def test_taking_them_away_does_not_depend_on_the_chat_still_having_a_name_of_its_own(self, typing_host, mac, chat):
        """A second group called the same appeared while the reply was worked on: no name finds
        the chat any more, and its window is still the host's to clear and close."""
        typing_host.typing("any;+;chat2", True)
        chat.chat(3, "any;+;chat3", [1, 2], name="Family", style=GROUP)
        assert typing_host.typing("any;+;chat2", False) == TypingResult()
        assert mac.closed == ["Family"] and mac.windows == [mac.main]

    def test_a_chat_the_host_cannot_name_gets_no_dots_and_messages_is_not_touched(self, typing_host, mac, chat):
        chat.chat(3, "any;+;chat3", [1, 2], style=GROUP)     # a group nobody named
        answer = typing_host.typing("any;+;chat3", True)
        assert (answer.showing, answer.reason) == (False, TYPING_NO_NAME) and answer.detail
        assert mac.calls == 0

    def test_health_names_the_gate_once_dots_were_asked_for_and_refused(self, typing_host, mac):
        assert typing_host.health().typing == ""
        mac.granted = False
        typing_host.typing("any;+;chat2", True)
        assert typing_host.health().typing == TYPING_NO_ACCESSIBILITY
        mac.granted = True
        assert typing_host.health().typing == ""

    def test_closing_the_host_leaves_no_window_and_no_draft_behind(self, typing_host, mac):
        typing_host.typing("any;+;chat2", True)
        typing_host.typing("any;-;+46701740605", True)
        typing_host.close()
        assert mac.windows == [mac.main] and sorted(mac.closed) == ["Alice Smith", "Family"]

    def test_whatever_goes_wrong_is_the_answer_never_an_exception(self, typing_host):
        with patch.object(typing_host.contacts, "sidebar_name", side_effect=OSError("AddressBook went away")):
            answer = typing_host.typing("any;+;chat2", True)
        assert (answer.showing, answer.reason) == (False, TYPING_FAILED) and "OSError" in answer.error
        with patch.object(typing_host.typing_indicator, "off", side_effect=RuntimeError("no element")):
            assert typing_host.typing("any;+;chat2", False).reason == TYPING_FAILED

    def test_a_host_that_never_shows_dots_never_loads_the_frameworks(self, host):
        """The conftest refuses the real loader: these calls would raise if they reached it."""
        assert host.health(refresh=True).typing == ""
        assert host.typing("any;+;chat2", False) == TypingResult()
        host.close()


@pytest.mark.unit
class TestAttachmentsCrossByReference:
    def test_the_wire_call_and_the_local_shortcut_give_the_same_bytes(self, host):
        (attachment,) = host.poll(1, None, 1).messages[0].attachments
        raw, reason = host.read_attachment(attachment.id)
        assert (raw, reason) == (b"\xff\xd8\xff fake jpeg \x00\x01", "")
        assert base64.b64decode(host.attachment(attachment.id).data) == raw

    def test_a_host_with_only_the_wire_call_still_hands_out_bytes(self, host):
        """What a remote host inherits: `read_attachment` built on `attachment` alone."""
        wire_only = Mock(spec=MessagesHost)
        wire_only.attachment.side_effect = host.attachment
        assert MessagesHost.read_attachment(wire_only, 40) == (b"\xff\xd8\xff fake jpeg \x00\x01", "")
        raw, reason = MessagesHost.read_attachment(wire_only, 41)
        assert raw is None and "41" in reason

    def test_an_unknown_reference_is_an_error_value(self, host):
        assert host.read_attachment(41)[0] is None
        answer = host.attachment(41)
        assert answer.data == "" and answer.error
        assert host.attachment_size(41).error

    def test_the_size_is_what_the_bytes_come_to(self, host):
        size = host.attachment_size(40)
        assert (size.value, size.error) == (len(host.read_attachment(40)[0]), "")

    def test_a_host_that_cannot_tell_the_size_says_so(self):
        """What a remote host inherits: no size without the bytes, as an error and never a zero
        a caller could take for an empty file."""
        answer = MessagesHost.attachment_size(Mock(spec=MessagesHost), 40)
        assert answer.error and answer.value == 0


@pytest.mark.unit
class TestFailuresCrossAsValues:
    """No sqlite or subprocess exception leaves the host: the other side of a wire has neither."""

    @pytest.fixture
    def broken(self, tmp_path):
        database = MessagesDatabase(tmp_path / "missing" / "chat.db")
        return LocalMessagesHost(database, Contacts(database, tmp_path / "AddressBook"))

    def test_health_names_the_reason(self, broken):
        health = broken.health(refresh=True)
        assert not health.ok and health.reason == NO_DATABASE and "chat.db" in health.detail

    def test_every_read_answers_with_an_error(self, broken):
        assert broken.cursor_at(NOW).error
        assert broken.last_rowid().error
        assert broken.remaining(0, NOW).error
        assert broken.scheduled().error
        assert broken.messages([1]).error
        assert broken.attachment(1).error

    def test_a_failed_poll_leaves_the_callers_cursor_where_it_was(self, broken):
        batch = broken.poll(17, None, 500)
        assert batch.error and batch.cursor == 17 and batch.messages == []

    def test_a_resolver_that_raises_is_an_error_value(self, host):
        host.contacts = Mock()
        host.contacts.resolve.side_effect = RuntimeError("AddressBook on fire")
        assert "AddressBook on fire" in host.resolve("Alice").error
        assert "AddressBook on fire" in host.send("Alice", "hi").error

    @pytest.mark.parametrize("step", ["refusal", "send"])
    def test_a_sender_that_raises_is_an_error_value_for_either_kind_of_send(self, host, step):
        """The rate limit is asked before anything else, and what it raised left the host."""
        host.sender = Mock(spec=sender_module.MessagesSender, refusal=Mock(return_value=""),
                           chat_target=sender_module.MessagesSender.chat_target)
        getattr(host.sender, step).side_effect = IndexError("pop from an empty deque")
        assert "IndexError" in host.send("+46701740605", "hi").error
        assert "IndexError" in host.send_chat("any;-;+46701740605", "hi").error

    def test_without_contacts_a_name_fails_and_a_raw_handle_still_sends(self, broken):
        assert broken.resolve("Alice").no_contacts
        with osascript():
            assert broken.send("+46701740605", "hi").error == ""


@pytest.mark.unit
class TestNoPerConsumerState:
    """Two consumers poll one host, each with its own cursor and horizon."""

    def test_two_consumers_read_one_host_without_seeing_each_other(self, host):
        feed_cursor, chat_cursor = 0, 2
        first = host.poll(feed_cursor, None, 2)
        other = host.poll(chat_cursor, NOW - 150, 500)
        assert [m.rowid for m in first.messages] == [1, 2]
        assert [m.rowid for m in other.messages] == [3, 4]
        again = host.poll(feed_cursor, None, 2)
        assert again == first, "a poll moves nothing in the host: the cursor is the caller's"
        assert host.poll(first.cursor, None, 500).messages == other.messages

    def test_the_rows_one_consumer_tracks_are_its_own_argument(self, host):
        assert [m.rowid for m in host.messages([4]).messages] == [4]
        assert [m.rowid for m in host.messages([1]).messages] == [1]
        assert [m.rowid for m in host.messages([4]).messages] == [4]

    def test_the_owner_is_the_callers_to_name(self, host):
        host.contacts.refresh()
        assert host.resolve("Marcus, Alice and Bob", owner="Marcus").handle == "any;+;chat2"
        assert host.resolve("Marcus, Alice and Bob", owner="Keep").handle is None

    def test_a_resolved_recipient_is_handed_out_not_kept(self, host):
        """The answer carries the group and the names; the host has no memory of what it resolved."""
        host.resolve("Family")
        kept = {name: value for part in (host, host.database, host.sender) for name, value in vars(part).items()}
        assert not any(isinstance(value, (Recipient, GroupChat)) or "chat2" in repr(value) for value in kept.values())


def crossing_types():
    """Every dataclass the contract module defines: what a call may take or return."""
    return [value for _, value in inspect.getmembers(contract, dataclasses.is_dataclass)]


def crosses(hint) -> bool:
    """Whether a value of this type is the same after a trip through JSON."""
    if hint in (str, int, float, bool):
        return True
    if get_origin(hint) is Union:
        return all(arg is type(None) or crosses(arg) for arg in get_args(hint))
    if get_origin(hint) is list:
        return crosses(get_args(hint)[0])
    return hint in crossing_types()


SAMPLES = [
    HostHealth(False, NO_DATABASE, "/Users/marcus/Library/Messages/chat.db does not exist", 12),
    HostHealth(True, contacts=12, satellite=SatelliteStatus(1, "v2026.11.7 (0a1b2c3d, release)", "release", "0a" * 20,
                                                            True, "Git pull failed: could not resolve host")),
    HostHealth(False, UNREACHABLE, "no connection (Connection refused)"),
    SatelliteStatus(2, "v2026.11.9 (5e5e5e5e, master, dirty)", "master", "5e" * 20),
    Count(41, "OperationalError: database is locked"),
    Attachment(40, "IMG_0001.jpg", "image/jpeg"),
    Message(7, NOW, True, "On my way", None, "", "Family", ["Alice", "Bob"],
            [Attachment(40, "IMG_0001.jpg", "image/jpeg"), Attachment(41, "clip.mov", None)], True, "completed", False),
    MessageBatch([Message(1, NOW, text="Hi", handle="+46701740605", sender="Alice"), Message(2)], 2, False, ""),
    Message(8, NOW, text="and dessert?", handle="+46701740605", sender="Alice", chat_name="Family", guid="G-8",
            chat_guid="any;+;chat2", is_group=True, service="iMessage", reply_to="OWN-1", reply_to_from_me=True,
            reply_to_text="Dinner is at seven"),
    Message(9, NOW, text='Loved "Dinner is at seven"', guid="G-9", chat_guid="any;-;+46701740605", tapback=2000),
    GroupChat("any;+;chat2", "Family", ["Alice", "Bob"], 700, ["+46701740605", "+46701740606"]),
    GroupList([GroupChat("any;+;chat2", "Family", ["Alice", "Bob"], 700, ["+46701740605"])], ""),
    GroupList(error="OperationalError: database is locked"),
    Recipient("any;+;chat2", "", None, None, GroupChat("any;+;chat2", None, ["Alice", "Bob"], 700), False),
    Recipient(None, "contact cache is empty — could not look up 'Alice'", no_contacts=True),
    SendResult("", Recipient("+46701740605", name="Alice", full_name="Alice Smith"), "sent:imessage", False),
    SendResult("not authorized to control Messages", unauthorized=True),
    AttachmentData(base64.b64encode(b"\x00\xff").decode(), ""),
    HostHealth(True, contacts=12, typing=TYPING_NO_ACCESSIBILITY),
    TypingResult(True),
    TypingResult(False, TYPING_NO_ROW, "no visible row of the Messages sidebar named 'Family'"),
    TypingResult(reason=TYPING_FAILED, error="iMessage satellite at http://keep-mini.local:8898: no connection"),
]


@pytest.mark.unit
class TestEverythingCrossesAsJson:
    def test_every_type_of_the_contract_has_a_sample(self):
        assert {type(sample) for sample in SAMPLES} == set(crossing_types())

    @pytest.mark.parametrize("value", SAMPLES, ids=lambda v: type(v).__name__)
    def test_a_value_survives_the_round_trip_through_json(self, value):
        assert from_wire(type(value), json.loads(json.dumps(to_wire(value)))) == value

    def test_what_a_real_host_answers_survives_it_too(self, host):
        host.health(refresh=True)
        with osascript():
            answers = [host.health(), host.cursor_at(NOW - 150), host.last_rowid(), host.remaining(0, None),
                       host.poll(0, None, 500),
                       host.messages([3, 4]), host.scheduled(), host.attachment(40), host.attachment(41),
                       host.resolve("Alice"), host.resolve("Family"), host.resolve("Zelda"),
                       host.send("Family", "dinner at 7"), host.send("Zelda", "hi"),
                       host.send_chat("any;+;chat2", "dinner at 7"), host.groups(),
                       host.typing("any;+;chat2", True), host.typing("any;+;chat2", False)]
        for answer in answers:
            assert from_wire(type(answer), json.loads(json.dumps(to_wire(answer)))) == answer

    def test_the_fields_are_plain_values_and_lists_of_them(self):
        """A tuple, a set, bytes or a Path would not come back from JSON as it went in."""
        for crossing in crossing_types():
            for name, hint in get_type_hints(crossing).items():
                assert crosses(hint), f"{crossing.__name__}.{name}: {hint}"

    def test_the_two_ends_may_be_a_version_apart(self):
        """A field the reader does not know is dropped; one the writer did not send keeps its default."""
        newer = {**to_wire(Message(7, NOW, text="Hi")), "edited_at": 1794000000.5}
        assert from_wire(Message, newer) == Message(7, NOW, text="Hi")
        assert from_wire(MessageBatch, {"messages": [{"rowid": 7}], "cursor": 7}) == MessageBatch([Message(7)], 7)

    def test_a_peer_from_before_the_chat_fields_is_read_as_a_message_in_no_known_chat(self):
        """A satellite one version behind sends no guid, chat, service or reply: the consumer gets
        the defaults — which a chat consumer reads as "not a group, not iMessage, no reply"."""
        older = {"rowid": 7, "timestamp": NOW, "is_from_me": False, "text": "Hi", "handle": "+46701740605",
                 "sender": "Alice", "chat_name": None, "recipients": [], "attachments": [], "send_later": False,
                 "schedule_state": "", "in_chat": True}
        message = from_wire(Message, json.loads(json.dumps(older)))
        assert (message.text, message.handle) == ("Hi", "+46701740605")
        assert (message.guid, message.chat_guid, message.is_group, message.service) == ("", "", False, "")
        assert (message.reply_to, message.reply_to_from_me, message.reply_to_text, message.tapback) == ("", False, None, 0)
        group = from_wire(GroupChat, {"guid": "any;+;chat2", "name": "Family", "members": ["Alice"], "last_active": 7})
        assert group.handles == []

    def test_every_call_of_the_interface_takes_and_returns_such_values(self):
        """The interface itself: arguments are plain, results are the contract's dataclasses —
        `read_attachment`, the one call that returns bytes, is built on `attachment`."""
        for name in sorted(MessagesHost.__abstractmethods__):
            hints = get_type_hints(getattr(MessagesHost, name))
            assert hints.pop("return") in crossing_types(), name
            for argument, hint in hints.items():
                assert crosses(hint), f"{name}({argument}: {hint})"


@pytest.mark.unit
class TestTheLayerStandsAlone:
    def test_the_host_layer_imports_nothing_of_the_assistant_the_interfaces_or_the_integrations(self):
        """A satellite serves this package without the assistant tree."""
        for path in sorted((SRC_DIR / "common" / "imessage").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else \
                    [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for module in modules:
                    assert module.split(".")[0] not in ("assistant", "interfaces", "integrations"), \
                        f"{path.name} imports {module}"
