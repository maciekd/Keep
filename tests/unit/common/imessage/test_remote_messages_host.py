"""RemoteMessagesHost against a real SatelliteService on localhost: every call of the interface
over HTTP, the attachment as bytes, and every way the other end can fail — as values."""
import base64
import json
import threading
import time
from unittest.mock import Mock, patch

import pytest
import requests

import common.imessage.RemoteMessagesHost as remote_module
import common.imessage.MessagesSender as sender_module
from common.imessage import Wire
from common.imessage.Contacts import Contacts
from common.imessage.LocalMessagesHost import LocalMessagesHost
from common.imessage.MessagesDatabase import MessagesDatabase
from common.imessage.MessagesHost import (BAD_ANSWER, Count, GroupChat, HostHealth, Message, MessageBatch,
                                          MessagesHost, NO_ACCESS, NOT_CONFIGURED, REFUSED, Recipient,
                                          SatelliteStatus, SendResult, TYPING_FAILED, TYPING_SCREEN_LOCKED,
                                          TypingResult, UNREACHABLE)
from common.imessage.RemoteMessagesHost import RemoteMessagesHost
from tests.unit.integrations.imessage.host_fakes import FakeHost, photo, received, send_later, sent

from .messages_fakes import ChatDb, GROUP, NOW, address_book
from .satellite_fakes import (SERVER_BRANCH, SERVER_COMMIT, TOKEN, canned_peer, closed_port, http_response,
                              server_identity, silent_peer, trickling_peer)

FAMILY = Recipient("any;+;chat2", name=None, group=GroupChat("any;+;chat2", "Family", ["Alice", "Bob"], 700))
ALICE = Recipient("+46701740605", name="Alice", full_name="Alice Smith")
JPEG = b"\xff\xd8\xff fake jpeg \x00\x01" * 50


def filled(host: FakeHost) -> FakeHost:
    host.rows = [received(1, "Hello"), received(2, "Look", attachments=[photo()]), sent(3, "On my way"),
                 received(4, "Dinner?", sender="Bob", handle="+46701740606", chat_name="Family")]
    host.tracked = [send_later(9, text="See you tomorrow")]
    host.scheduled_rows = [send_later(9), send_later(10, state="failed", in_chat=False)]
    host.known = {"alice": ALICE, "family": FAMILY}
    host.group_chats = [GroupChat("any;+;chat2", "Family", ["Alice", "Bob"], 700, ["+46701740605", "+46701740606"])]
    host.files = {40: JPEG}
    host.start = 17
    return host


CALLS = [
    ("health", (True,)), ("health", ()),
    ("cursor_at", (NOW - 3600,)), ("cursor_at", (None,)),
    ("last_rowid", ()),
    ("remaining", (2, None)),
    ("poll", (0, None, 500)), ("poll", (1, NOW - 3600.5, 2)),
    ("messages", ([9, 11],)), ("messages", ([],)),
    ("scheduled", ()),
    ("resolve", ("Alice",)), ("resolve", ("Family", "Marcus")), ("resolve", ("Zelda",)),
    ("send", ("Alice", "Kör försiktigt 🚗")), ("send", ("Family", "dinner at 7", "Marcus")), ("send", ("Zelda", "hi")),
    ("send_chat", ("any;+;chat2", "Kör försiktigt 🚗")), ("send_chat", ("m@mlab.test", "hi")),
    ("groups", ()),
    ("typing", ("any;+;chat2", True)), ("typing", ("any;-;+46701740605", False)),
]


@pytest.mark.unit
class TestEveryCallCrosses:
    @pytest.mark.parametrize("name, arguments", CALLS, ids=[f"{name}-{n}" for n, (name, _) in enumerate(CALLS)])
    def test_the_answer_is_what_the_host_behind_the_satellite_answers(self, remote, fake, follower, name, arguments):
        """The same call on a twin of the host, in this process, is the measure."""
        twin = getattr(filled(FakeHost()), name)(*arguments)
        answer = getattr(remote, name)(*arguments)
        if name == "health":
            assert answer.satellite == follower.answer
            answer.satellite = None
        assert answer == twin

    def test_the_host_is_asked_with_the_callers_arguments(self, remote, fake):
        remote.health(refresh=True)
        remote.poll(41, 1794000000.5, 200)
        remote.messages([9, 11])
        remote.send("Alice", "hi", "Marcus")
        assert fake.calls == [("health", True), ("poll", 41, 1794000000.5, 200), ("messages", [9, 11]),
                              ("send", "Alice", "hi", "Marcus"), ("resolve", "Alice", "Marcus")]

    def test_the_calls_the_test_above_makes_are_all_the_interface_has(self):
        assert {name for name, _ in CALLS} == set(MessagesHost.__abstractmethods__) - {"attachment"}

    @pytest.fixture(autouse=True)
    def a_filled_host(self, fake):
        filled(fake)

    def test_a_real_local_host_behind_the_satellite_reads_the_same_as_in_process(self, remote, satellite, tmp_path):
        """chat.db, the AddressBook and a file on disk at one end, the consumer at the other."""
        db = ChatDb(tmp_path / "chat.db")
        db.handle(1, "+46701740605").handle(2, "+46701740606")
        db.chat(1, "any;-;+46701740605", [1]).chat(2, "any;+;chat2", [1, 2], name="Family", style=GROUP)
        picture = tmp_path / "IMG_0001.jpg"
        picture.write_bytes(JPEG)
        db.message(1, "Hello", ts=NOW - 300, chat=1, sender=1)
        db.message(2, "Look", ts=NOW - 200, chat=2, sender=2).attachment(40, 2, str(picture), "image/jpeg")
        database = MessagesDatabase(db.path)
        people = [("Alice", "Smith", [("070-174 06 05", "_$!<Mobile>!$_")], [])]
        local = LocalMessagesHost(database, Contacts(database, address_book(tmp_path, people)))
        satellite.service.host = local
        sent_ok = Mock(returncode=0, stdout="sent:imessage", stderr="")

        assert remote.health(refresh=True).contacts == local.health().contacts > 0
        assert remote.poll(0, None, 500) == local.poll(0, None, 500)
        assert remote.resolve("Family") == local.resolve("Family")
        assert remote.read_attachment(40) == (JPEG, "")
        with patch.object(sender_module.subprocess, "run", return_value=sent_ok) as osascript:
            answer = remote.send("Alice", "On my way")
        handle = local.resolve("Alice").handle
        script = osascript.call_args.args[0][-1]
        assert answer.error == "" and answer.recipient.handle == handle
        assert handle in script and "On my way" in script

    def test_a_poll_carries_what_a_chat_consumer_reads_across_the_wire(self, remote, fake):
        fake.rows = [Message(8, NOW, text="and dessert?", handle="+46701740605", sender="Alice", chat_name="Family",
                             guid="G-8", chat_guid="any;+;chat2", is_group=True, service="iMessage",
                             reply_to="OWN-1", reply_to_from_me=True, reply_to_text="Dinner is at seven")]
        (message,) = remote.poll(0, None, 10).messages
        assert message == fake.rows[0]

    def test_the_dots_cost_one_request_each_way_and_the_hosts_verdict_comes_back(self, remote, fake):
        """The keep-alive is the host's: the wire carries the `on` and the `off`, nothing between."""
        fake.calls.clear()
        assert remote.typing("any;+;chat2", True) == TypingResult(showing=True)
        fake.typing_answer = TypingResult(reason=TYPING_SCREEN_LOCKED, detail="the screen is locked")
        assert remote.typing("any;+;chat2", True) == fake.typing_answer
        fake.typing_answer = None
        assert remote.typing("any;+;chat2", False) == TypingResult()
        assert fake.calls == [("typing", "any;+;chat2", True), ("typing", "any;+;chat2", True),
                              ("typing", "any;+;chat2", False)]

    def test_the_gate_that_keeps_the_dots_away_comes_with_the_satellites_health(self, remote, fake):
        fake.health_answer = HostHealth(True, contacts=3, typing=TYPING_SCREEN_LOCKED)
        assert remote.health().typing == TYPING_SCREEN_LOCKED

    def test_a_satellite_from_before_the_dots_answers_with_a_value_and_the_reply_goes_on(self):
        with canned_peer(http_response("404 NOT FOUND", b"{}", X_Keep_Wire="1")) as (address, _), server_identity():
            answer = RemoteMessagesHost(address, TOKEN).typing("any;+;chat2", True)
        assert (answer.showing, answer.reason) == (False, TYPING_FAILED) and "typing" in answer.error

    def test_a_chat_send_the_satellite_never_answered_may_have_gone_out(self, fake):
        """As for `send`: the caller is told, so the same bubble is not sent twice blindly."""
        with silent_peer() as address, server_identity(), patch.object(remote_module, "READ_TIMEOUTS", {"send_chat": 0.03}):
            answer = RemoteMessagesHost(address, TOKEN).send_chat("any;+;chat2", "hi")
        assert "may have been sent" in answer.error

    def test_a_satellite_from_before_the_chat_calls_answers_with_an_error_not_an_exception(self):
        """One version behind: the satellite knows neither `send_chat` nor `groups` and answers 404."""
        with canned_peer(http_response("404 NOT FOUND", b"{}", X_Keep_Wire="1")) as (address, _), server_identity():
            host = RemoteMessagesHost(address, TOKEN)
            sent, listing = host.send_chat("any;+;chat2", "hi"), host.groups()
        assert "send_chat" in sent.error and "may have been sent" not in sent.error
        assert listing.groups == [] and "groups" in listing.error


@pytest.mark.unit
class TestAttachmentsCrossAsBytes:
    def test_the_bytes_arrive_as_they_are_on_the_hosts_disk(self, remote, fake):
        fake.files[40] = JPEG
        assert remote.read_attachment(40) == (JPEG, "")

    def test_the_interfaces_base64_call_is_built_on_this_side(self, remote, fake):
        """`attachment` still answers, for a caller that wants it — encoded after the bytes crossed."""
        fake.files[40] = JPEG
        assert base64.b64decode(remote.attachment(40).data) == JPEG

    def test_a_file_the_host_does_not_have_is_an_error_naming_it(self, remote):
        raw, reason = remote.read_attachment(41)
        assert raw is None and "41" in reason
        assert remote.attachment(41).data == "" and "41" in remote.attachment(41).error

    def test_the_satellite_does_not_send_a_file_over_the_cap(self, remote, fake):
        fake.files[40] = JPEG
        with patch.object(Wire, "MAX_ATTACHMENT_BYTES", len(JPEG) - 1):
            raw, reason = remote.read_attachment(40)
        assert raw is None and "40" in reason

    def test_this_side_stops_reading_an_answer_over_its_cap(self, remote, fake):
        """A peer that sends more than it should is not read to the end, whatever it declares."""
        fake.rows = [received(n, "x" * 200) for n in range(1, 40)]
        with patch.object(Wire, "MAX_ANSWER_BYTES", 2000):
            batch = remote.poll(0, None, 500)
        assert batch.messages == [] and batch.error and batch.cursor == 0
        body = json.dumps({"messages": [], "cursor": 3}).encode() + b" " * 5000
        undeclared = http_response("200 OK", body, X_Keep_Wire="1").replace(f"Content-Length: {len(body)}\r\n".encode(), b"")
        with canned_peer(undeclared) as (address, _), server_identity(), patch.object(Wire, "MAX_ANSWER_BYTES", 2000):
            unbounded = RemoteMessagesHost(address, TOKEN).poll(7, None, 500)
        assert unbounded.cursor == 7 and unbounded.error


@pytest.mark.unit
class TestTheServerSaysWhatItRuns:
    def test_every_request_names_the_branch_and_the_commit(self, remote, fake, follower):
        fake.files[40] = JPEG
        remote.health(), remote.poll(0, None, 10), remote.read_attachment(40), remote.send("Alice", "hi")
        assert follower.notices == [(SERVER_BRANCH, SERVER_COMMIT)] * 4

    @pytest.mark.parametrize("branch, commit", [(None, SERVER_COMMIT), ("release", "")], ids=["detached", "no git"])
    def test_a_server_that_cannot_tell_names_neither(self, satellite, follower, branch, commit):
        with server_identity(branch, commit):
            host = RemoteMessagesHost(satellite.address, TOKEN)
        assert host.health().ok
        assert follower.notices == [("", "")]

    def test_health_brings_what_the_satellite_says_of_itself(self, remote, follower):
        follower.answer = SatelliteStatus(1, "v2026.10.3 (7f7f7f7f, master)", "master", "7f" * 20, True,
                                          "Git pull failed: could not resolve host")
        assert remote.health().satellite == follower.answer

    def test_a_host_in_this_process_has_no_satellite_and_no_address(self, tmp_path):
        local = LocalMessagesHost(MessagesDatabase(tmp_path / "none.db"))
        assert local.health().satellite is None and local.address == ""


@pytest.mark.unit
class TestTheWireVersion:
    def test_a_satellite_one_version_ahead_works_and_its_version_is_reported(self, remote, satellite, fake):
        satellite.service.wire = Wire.WIRE_VERSION + 1
        fake.rows = [received(1, "Hello")]
        assert [m.text for m in remote.poll(0, None, 10).messages] == ["Hello"]
        health = remote.health()
        assert health.ok and health.satellite.wire == Wire.WIRE_VERSION + 1

    def test_a_satellite_two_versions_away_is_an_error_naming_both(self, remote, satellite, fake):
        satellite.service.wire = Wire.WIRE_VERSION + 2
        health = remote.health()
        assert (health.ok, health.reason) == (False, BAD_ANSWER)
        assert str(Wire.WIRE_VERSION + 2) in health.detail and str(Wire.WIRE_VERSION) in health.detail
        assert remote.poll(5, None, 10).error and fake.calls == [], "the calls of a caller too far away are not run"

    def test_a_wire_too_far_apart_says_which_end_is_behind(self, remote, satellite):
        """The older end is the one to update: a satellite follows its server by itself, a
        server behind its satellite is not helped by anything done on the satellite's Mac."""
        satellite.service.wire = Wire.WIRE_VERSION + 2
        newer = remote.health()
        assert newer.satellite.wire == Wire.WIRE_VERSION + 2
        satellite.service.wire = Wire.WIRE_VERSION
        with patch.object(Wire, "WIRE_VERSION", Wire.WIRE_VERSION + 2):
            older = remote.health()
            assert older.satellite.wire == Wire.WIRE_VERSION - 2
        assert not newer.ok and not older.ok
        waits_for = [health.detail.rsplit("until", 1)[1] for health in (newer, older)]
        assert "server" in waits_for[0] and "satellite" not in waits_for[0]
        assert "satellite" in waits_for[1] and "server" not in waits_for[1]

    def test_a_satellite_too_far_away_is_waited_for_since_it_is_updating_itself(self, remote, satellite, fake):
        """It follows this server at that very request: a consumer that counted the refusals
        would have given up before the satellite is back on a version it can talk to."""
        satellite.service.wire = Wire.WIRE_VERSION + 2
        assert remote.poll(5, None, 10).error and remote.away()
        satellite.service.wire = Wire.WIRE_VERSION
        fake.error = "OperationalError: database is locked"
        assert remote.poll(5, None, 10).error and not remote.away(), "a failure of its own is still a failure"

    def test_a_field_this_side_does_not_know_is_dropped_and_a_missing_one_defaulted(self):
        newer = json.dumps({"messages": [{"rowid": 7, "text": "Hi", "edited_at": 1794000000.5}], "cursor": 7}).encode()
        with canned_peer(http_response("200 OK", newer, X_Keep_Wire="2")) as (address, _), server_identity():
            batch = RemoteMessagesHost(address, TOKEN).poll(0, None, 10)
        assert (batch.error, batch.cursor, batch.exhausted) == ("", 7, True)
        assert [(m.rowid, m.text, m.in_chat) for m in batch.messages] == [(7, "Hi", True)]

    def test_a_call_the_satellite_does_not_know_is_an_error_not_an_exception(self):
        unknown = http_response("404 NOT FOUND", b'{"error": "the satellite does not know the call \'resolve\'"}', X_Keep_Wire="0")
        with canned_peer(unknown) as (address, _), server_identity():
            answer = RemoteMessagesHost(address, TOKEN).resolve("Alice")
        assert answer.handle is None and "resolve" in answer.error


@pytest.mark.unit
class TestEveryFailureIsAValue:
    @pytest.fixture
    def unreachable(self):
        with server_identity():
            return RemoteMessagesHost(closed_port(), TOKEN)

    def test_nothing_listening_fails_every_call_in_its_own_type(self, unreachable):
        health = unreachable.health(refresh=True)
        assert (health.ok, health.reason, health.satellite) == (False, UNREACHABLE, None)
        answers = [unreachable.cursor_at(None), unreachable.remaining(3, None), unreachable.messages([1]),
                   unreachable.scheduled(), unreachable.resolve("Alice"), unreachable.send("Alice", "hi"),
                   unreachable.attachment(40)]
        assert [type(a) for a in answers[:6]] == [Count, Count, MessageBatch, MessageBatch, Recipient, SendResult]
        assert all(a.error for a in answers)
        assert unreachable.read_attachment(40)[0] is None and unreachable.read_attachment(40)[1]

    def test_dots_asked_of_a_satellite_that_is_not_there_are_no_dots_not_an_exception(self, unreachable):
        answer = unreachable.typing("any;+;chat2", True)
        assert (answer.showing, answer.reason) == (False, TYPING_FAILED) and answer.error

    def test_only_a_host_nobody_answers_for_is_away(self, unreachable, remote, satellite, fake):
        """What a polling consumer waits out, as opposed to what it counts as a failure."""
        assert not unreachable.away() and not remote.away(), "nothing was asked yet"
        unreachable.poll(0, None, 10)
        fake.error = "OperationalError: database is locked"
        assert remote.poll(0, None, 10).error
        assert unreachable.away() and not remote.away()
        with server_identity():
            refused, unset = RemoteMessagesHost(satellite.address, "another-token-entirely"), RemoteMessagesHost(satellite.address, "")
        refused.health(), unset.health()
        assert not refused.away() and not unset.away(), "a wrong or missing token does not mend itself"
        remote.address = unreachable.address
        remote.health()
        assert remote.away()

    def test_a_failed_poll_leaves_the_callers_cursor_where_it_was(self, unreachable):
        batch = unreachable.poll(4711, None, 500)
        assert (batch.cursor, batch.messages) == (4711, []) and batch.error

    def test_a_send_that_never_connected_was_not_sent(self, unreachable):
        assert "may have been sent" not in unreachable.send("Alice", "hi").error

    def test_no_answer_in_time_is_unreachable_and_a_send_may_have_gone_out(self):
        with silent_peer() as address, server_identity(), patch.object(remote_module, "READ_TIMEOUT", 0.03), \
                patch.object(remote_module, "READ_TIMEOUTS", {}):
            host = RemoteMessagesHost(address, TOKEN)
            health, result = host.health(), host.send("Alice", "hi")
        assert (health.ok, health.reason) == (False, UNREACHABLE)
        assert "may have been sent" in result.error

    HEAD = b"HTTP/1.1 200 OK\r\nX-Keep-Wire: 1\r\nContent-Length: 4000\r\n\r\n"

    @pytest.mark.parametrize("at_once, drip", [(b"", HEAD), (HEAD, b" " * 4000)], ids=["the headers", "the body"])
    def test_a_peer_that_sends_a_byte_now_and_then_does_not_hold_the_caller_past_the_deadline(self, at_once, drip):
        """No single read waits long enough to time out, so only a deadline for the whole call ends it."""
        with trickling_peer(at_once, drip) as (address, _), server_identity(), \
                patch.object(remote_module, "READ_TIMEOUT", 1.0), patch.object(remote_module, "READ_TIMEOUTS", {}), \
                patch.object(remote_module, "TRANSFER_SECONDS", -0.98):
            host = RemoteMessagesHost(address, TOKEN)
            started = time.monotonic()
            batch, sent_ = host.poll(8, None, 10), host.send("Alice", "hi")
            took = time.monotonic() - started
        assert batch.error and batch.cursor == 8 and host.away()
        assert "may have been sent" in sent_.error
        assert took < 1.0, "each call was given 20 ms in all; the peer would have gone on for 40 s"

    def test_a_peer_that_never_finishes_costs_no_thread_per_call(self):
        """An exchange given up on still waits for its peer: after a few of them nothing more is sent."""
        with trickling_peer(b"", b"x" * 4000) as (address, connections), server_identity(), \
                patch.object(remote_module, "READ_TIMEOUT", 1.0), patch.object(remote_module, "READ_TIMEOUTS", {}), \
                patch.object(remote_module, "TRANSFER_SECONDS", -0.98):
            host = RemoteMessagesHost(address, TOKEN)
            answers = [host.cursor_at(None) for _ in range(remote_module.MAX_ABANDONED + 4)]
            assert all(answer.error for answer in answers) and host.away()
            waited = time.monotonic() + 2
            while len(connections) < remote_module.MAX_ABANDONED and time.monotonic() < waited:
                time.sleep(0.005)
            assert len(connections) == remote_module.MAX_ABANDONED

    @pytest.mark.parametrize("body", [b"[" * 100_000 + b"]" * 100_000,
                                      b'{"messages": ' + b"[" * 100_000 + b"]" * 100_000 + b"}"],
                             ids=["nothing but brackets", "inside the rows"])
    def test_json_nested_deeper_than_python_reads_is_a_bad_answer_not_an_exception(self, body):
        """`json.loads` raises RecursionError for it, which is no ValueError."""
        refusal = http_response("500 INTERNAL SERVER ERROR", body, X_Keep_Wire="1")
        with canned_peer(http_response("200 OK", body, X_Keep_Wire="1")) as (address, _), server_identity():
            host = RemoteMessagesHost(address, TOKEN)
            batch, health, sent_ = host.poll(11, None, 10), host.health(), host.send("Alice", "hi")
        with canned_peer(refusal) as (address, _), server_identity():
            refused = RemoteMessagesHost(address, TOKEN).poll(11, None, 10)
        assert (batch.messages, batch.cursor) == ([], 11) and batch.error and refused.error
        assert (health.ok, health.reason) == (False, BAD_ANSWER) and sent_.error

    def test_the_verdict_on_a_call_is_its_callers_and_not_the_last_threads(self, remote, fake):
        """A poll that found nobody there is waited out; a send that went through on another
        thread in between must not turn it into a counted failure — nor the reverse."""
        with patch.object(remote._session, "request", side_effect=requests.ConnectionError("Connection refused")):
            assert remote.poll(0, None, 10).error
        seen_elsewhere = []
        other = threading.Thread(target=lambda: seen_elsewhere.extend([remote.health().ok, remote.away()]))
        other.start()
        other.join(5)
        assert seen_elsewhere == [True, False]
        assert remote.away(), "this thread's poll is the one that found nobody there"

    @pytest.mark.parametrize("response", [
        http_response("200 OK", b"{}"),
        http_response("200 OK", b"<html>router</html>", X_Keep_Wire="1"),
        http_response("200 OK", b"[1, 2]", X_Keep_Wire="1"),
        http_response("502 Bad Gateway", b"upstream down", X_Keep_Wire="1"),
        http_response("200 OK", b"{}", X_Keep_Wire="one"),
        http_response("200 OK", b"{}", X_Keep_Wire="9" * 5000),
        http_response("200 OK", b'{"ok": "yes", "value": "many", "contacts": [1]}', X_Keep_Wire="1"),
    ], ids=["no wire version", "not JSON", "not an object", "a failing gateway", "a wire version that is no number",
            "a wire version no int holds", "fields of other types"])
    def test_an_answer_that_is_not_a_satellites_is_a_bad_answer(self, response):
        with canned_peer(response) as (address, _), server_identity():
            host = RemoteMessagesHost(address, TOKEN)
            health, count = host.health(), host.cursor_at(None)
        assert (health.ok, health.reason) == (False, BAD_ANSWER) and health.detail
        assert count.error

    @pytest.mark.parametrize("body", [b'{"messages": "x"}', b'{"messages": [7]}', b'{"messages": [{"rowid": "7"}]}',
                                      b'{"messages": [{"rowid": 7, "attachments": [{"id": null}]}]}', b'{"cursor": "9"}',
                                      b'{"messages": [{"text": "Hi"}]}'],
                             ids=["a string for the rows", "a number for a row", "a string for a ROWID",
                                  "an attachment without an id", "a string for the cursor", "a row without a ROWID"])
    def test_rows_of_another_shape_are_an_error_and_the_cursor_stays(self, body):
        """What a consumer does with a row must not depend on what a stranger at the address answered."""
        with canned_peer(http_response("200 OK", body, X_Keep_Wire="1")) as (address, _), server_identity():
            batch = RemoteMessagesHost(address, TOKEN).poll(11, None, 10)
        assert (batch.messages, batch.cursor) == ([], 11) and batch.error

    def test_an_answer_cannot_take_the_callers_cursor_back(self):
        """Followed, it would have the caller read its history again from there."""
        older = http_response("200 OK", b'{"messages": [], "cursor": 1}', X_Keep_Wire="1")
        with canned_peer(older) as (address, _), server_identity():
            batch = RemoteMessagesHost(address, TOKEN).poll(500, None, 10)
        assert batch.cursor == 500 and not batch.error

    @pytest.mark.parametrize("body", [b"{}", b'{"error": ""}', b'{"messages": []}'],
                             ids=["an empty object", "nothing but no error", "rows without a cursor"])
    def test_an_object_without_what_every_answer_carries_is_no_answer(self, body):
        """Every field of a result has a default, so `{}` would read as a call that went well:
        a text reported as sent that never was, a table whose end is ROWID 0 — which the feed
        takes for another database and reads five years again."""
        with canned_peer(http_response("200 OK", body, X_Keep_Wire="1")) as (address, _), server_identity():
            host = RemoteMessagesHost(address, TOKEN)
            answers = [host.send("Alice", "hi"), host.send_chat("any;+;chat2", "hi"), host.last_rowid(),
                       host.cursor_at(None), host.poll(500, None, 10), host.resolve("Alice"), host.groups()]
            health, dots = host.health(), host.typing("any;+;chat2", True)
        assert all(answer.error for answer in answers)
        assert (health.ok, health.reason) == (False, BAD_ANSWER)
        assert not dots.showing and dots.reason == TYPING_FAILED
        assert answers[4].cursor == 500 and not host.away(), "a peer that answers is not one to wait for"

    def test_a_host_that_raises_behind_the_satellite_is_an_error_without_its_text(self, remote, satellite):
        """What a satellite's exception says may quote a message; only its type crosses."""
        satellite.service.host = Mock(spec=MessagesHost, poll=Mock(side_effect=RuntimeError("near 'Hemligt möte kl 9'")))
        batch = remote.poll(3, None, 10)
        assert batch.cursor == 3 and "RuntimeError" in batch.error and "Hemligt" not in batch.error

    def test_a_refused_token_is_told_apart_from_a_satellite_that_is_down(self, satellite):
        with server_identity():
            host = RemoteMessagesHost(satellite.address, TOKEN[::-1])
        assert host.health().reason == REFUSED
        assert host.poll(0, None, 10).error

    def test_a_redirect_is_not_followed_so_the_token_goes_nowhere_else(self):
        with canned_peer(http_response("200 OK", b"{}", X_Keep_Wire="1")) as (elsewhere, seen):
            moved = http_response("307 Temporary Redirect", b"", Location=f"{elsewhere}/messages/health", X_Keep_Wire="1")
            with canned_peer(moved) as (address, _), server_identity():
                health = RemoteMessagesHost(address, TOKEN).health()
        assert not health.ok and seen == []

    def test_the_satellites_own_verdict_on_its_database_crosses_untouched(self, remote, fake):
        fake.health_answer = HostHealth(False, NO_ACCESS, "cannot read /Users/keep/Library/Messages/chat.db", 0)
        health = remote.health()
        assert (health.ok, health.reason, health.detail) == (False, NO_ACCESS, fake.health_answer.detail)
        assert health.satellite is not None, "it answered: the problem is on its Mac, not on the way there"


@pytest.mark.unit
class TestAnAddressOrTokenThatCannotWork:
    @pytest.mark.parametrize("address, token", [
        ("192.0.2.7:8898", TOKEN), ("ftp://192.0.2.7:8898", TOKEN), ("http://", TOKEN), ("", TOKEN),
        (f"http://keep:{TOKEN}@192.0.2.7:8898", TOKEN), (f"http://192.0.2.7:8898/?token={TOKEN}", TOKEN),
        ("http://192.0.2.7:8898/messages", TOKEN), ("http://192.0.2.7:port", TOKEN),
        ("http://192.0.2.7:8898", ""), ("http://192.0.2.7:8898", "   "),
        ("http://192.0.2.7:8898", f"{TOKEN[:16]} {TOKEN[16:]}"), ("http://192.0.2.7:8898", f"{TOKEN}\r\nX-Keep-Branch: master"),
        ("http://192.0.2.7:8898", "lösenord-som-inte-är-en-token"),
    ], ids=["no scheme", "not http", "no host", "empty", "a login in it", "a query in it", "a path in it",
            "a port that is no number", "no token", "a blank token", "a space in the token", "a header smuggled in the token",
            "a token that is not ASCII"])
    def test_nothing_is_sent_and_the_reason_does_not_quote_what_was_typed(self, address, token):
        session = Mock(headers={})
        with server_identity():
            host = RemoteMessagesHost(address, token, session=session)
        health = host.health()
        assert (health.ok, health.reason) == (False, NOT_CONFIGURED)
        assert host.poll(3, None, 10).cursor == 3 and host.send("Alice", "hi").error
        assert host.read_attachment(40)[0] is None
        session.request.assert_not_called()
        assert TOKEN not in health.detail and TOKEN not in host.send("Alice", "hi").error
        assert TOKEN not in host.address, "a consumer words its hints with the address"

    def test_a_token_pasted_with_the_line_break_after_it_is_the_token(self, satellite):
        with server_identity():
            assert RemoteMessagesHost(satellite.address, f"  {TOKEN}\n").health().ok

    @pytest.mark.parametrize("address", ["http://192.0.2.7:8898", "http://192.0.2.7:8898/", " https://mac-mini.local:8898 ",
                                         "http://[2001:db8::7]:8898"])
    def test_an_address_of_a_host_and_a_port_is_taken(self, address):
        session = Mock(headers={})
        session.request.side_effect = remote_module.requests.ConnectionError("down")
        with server_identity():
            host = RemoteMessagesHost(address, TOKEN, session=session)
        assert host.health().reason == UNREACHABLE
        assert session.request.call_args.args[1] == f"{address.strip().rstrip('/')}/messages/health"


@pytest.mark.unit
class TestTheTokenStaysOutOfSight:
    def test_it_is_in_no_url_no_log_line_and_no_error(self, satellite, fake):
        logged = []
        with patch("common.logging.log", side_effect=lambda message, *a, **k: logged.append(str(message))), \
                server_identity():
            good = RemoteMessagesHost(satellite.address, TOKEN)
            refused = RemoteMessagesHost(satellite.address, TOKEN + "x")
            down = RemoteMessagesHost(closed_port(), TOKEN)
            answers = [host.health(True) for host in (good, refused, down)]
            answers += [host.send("Alice", "hi") for host in (good, refused, down)]
            answers += [host.read_attachment(40) for host in (good, refused, down)]
        assert logged, "the exchange is logged at all"
        assert TOKEN not in "\n".join(logged) and TOKEN not in repr(answers)
        assert TOKEN not in good.address and TOKEN not in repr(good) and TOKEN not in repr(vars(good))

    def test_the_request_carries_it_in_a_header_and_the_url_is_the_route_alone(self):
        with canned_peer(http_response("200 OK", b"{}", X_Keep_Wire="1")) as (address, seen), server_identity():
            RemoteMessagesHost(address, TOKEN).health()
        request_line, _, headers = seen[0].decode().partition("\r\n")
        assert request_line == "POST /messages/health HTTP/1.1"
        assert f"Authorization: Bearer {TOKEN}\r\n" in headers
