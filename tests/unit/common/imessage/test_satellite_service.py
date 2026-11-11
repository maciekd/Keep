"""SatelliteService as a stranger, a careless caller and a caller of another version meet it:
plain HTTP requests against the real app on localhost, a fake MessagesHost behind it."""
import asyncio
from unittest.mock import patch

import pytest
import requests
from werkzeug.exceptions import NotFound

import common.imessage.SatelliteService as service_module
from common.imessage import Wire
from common.imessage.MessagesHost import MessagesHost, Recipient
from tests.unit.integrations.imessage.host_fakes import received

from .satellite_fakes import TOKEN

AUTHORIZED = {"Authorization": f"Bearer {TOKEN}", "X-Keep-Wire": str(Wire.WIRE_VERSION)}


def post(satellite, call, body=None, headers=AUTHORIZED, **kwargs):
    return requests.post(f"{satellite.address}/messages/{call}", json=body, headers=headers, timeout=5, **kwargs)


@pytest.mark.unit
class TestNothingWithoutTheToken:
    STRANGERS = [{}, {"Authorization": f"Bearer {TOKEN[:-1]}"}, {"Authorization": f"Bearer {TOKEN}x"},
                 {"Authorization": TOKEN}, {"Authorization": f"Basic {TOKEN}"}, {"X-Keep-Token": TOKEN},
                 {"Authorization": "Bearer "}, {"Authorization": "Bearer lösenord-som-inte-är-en-token"}]
    IDS = ["no header", "one character short", "one too many", "no scheme", "another scheme", "another header", "empty",
           "not ASCII"]

    @pytest.mark.parametrize("headers", STRANGERS, ids=IDS)
    def test_every_route_answers_401_and_the_host_is_never_asked(self, satellite, fake, follower, headers):
        fake.files[40] = b"\xff\xd8 private photo"
        fake.rows = [received(1, "Koden till larmet är 4711")]
        headers = {**headers, "X-Keep-Wire": "1", "X-Keep-Branch": "master", "X-Keep-Commit": "ab" * 20}
        answers = [post(satellite, "poll", {"after_rowid": 0, "horizon_ts": None, "limit": 10}, headers),
                   post(satellite, "send", {"recipient": "Alice", "body": "hi"}, headers),
                   post(satellite, "health", {}, headers),
                   post(satellite, "no_such_call", {}, headers),
                   requests.get(f"{satellite.address}/messages/attachment/40", headers=headers, timeout=5),
                   requests.get(f"{satellite.address}/", headers=headers, timeout=5)]
        assert [a.status_code for a in answers] == [401] * 6
        assert fake.calls == [], "no call of the host ran"
        assert follower.notices == [], "a stranger cannot make the satellite update"
        assert all("4711" not in a.text and b"private" not in a.content for a in answers)

    def test_a_stranger_is_not_told_what_answers_here(self, satellite):
        answer = post(satellite, "health", {}, headers={})
        assert Wire.WIRE_HEADER not in answer.headers and "hypercorn" not in answer.headers.get("server", "")
        assert set(answer.json()) == {"error"} and "satellite" not in answer.text.lower()

    UPGRADE = {"Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Version": "13",
               "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="}

    @pytest.mark.parametrize("path", ["/messages/health", "/messages/attachment/40", "/nowhere"])
    def test_a_websocket_upgrade_is_refused_like_any_request_and_reaches_nothing(self, satellite, fake, follower, path):
        """An upgrade never passes `before_request`: without a gate of its own it reached the
        routing, and the error handlers failed on it."""
        fake.files[40] = b"\xff\xd8 private photo"
        stranger = requests.get(f"{satellite.address}{path}", headers=self.UPGRADE, timeout=5)
        server = requests.get(f"{satellite.address}{path}", headers={**self.UPGRADE, **AUTHORIZED}, timeout=5)
        assert stranger.status_code == 401 and Wire.WIRE_HEADER not in stranger.headers
        assert server.status_code == 400
        assert fake.calls == [] and follower.notices == []
        assert b"private" not in stranger.content + server.content

    def test_the_token_in_the_url_opens_nothing(self, satellite, fake):
        answer = requests.post(f"{satellite.address}/messages/scheduled?token={TOKEN}", json={}, timeout=5)
        assert answer.status_code == 401 and fake.calls == []

    def test_the_comparison_takes_the_same_time_wherever_the_token_differs(self, satellite):
        """hmac.compare_digest, not ==: a guess is not told how many characters it got right."""
        with patch.object(service_module.hmac, "compare_digest", wraps=service_module.hmac.compare_digest) as compare:
            post(satellite, "scheduled", {}, {"Authorization": "Bearer guess"})
            post(satellite, "scheduled", {})
        assert [call.args for call in compare.call_args_list] == [
            (b"Bearer guess", f"Bearer {TOKEN}".encode()), (f"Bearer {TOKEN}".encode(), f"Bearer {TOKEN}".encode())]

    def test_a_refusal_is_logged_without_what_was_presented(self, satellite):
        guess = "Bearer 9xGuessedWrongButCloseToARealOne42"
        logged = []
        with patch("common.logging.log", side_effect=lambda message, *a, **k: logged.append(str(message))):
            post(satellite, "health", {}, {"Authorization": guess})
        assert any("refused" in line for line in logged)
        assert not any("9xGuessed" in line or TOKEN in line for line in logged)


@pytest.mark.unit
class TestTheWireVersionOfTheCaller:
    def test_a_caller_that_names_none_is_refused(self, satellite, fake):
        answer = post(satellite, "scheduled", {}, {"Authorization": f"Bearer {TOKEN}"})
        assert answer.status_code == 400 and fake.calls == []

    @pytest.mark.parametrize("wire", ["one", "-1", "1.0", "9" * 5000, "", "²"])
    def test_a_wire_version_that_is_no_number_is_refused_not_a_crash(self, satellite, fake, wire):
        answer = post(satellite, "scheduled", {}, {**AUTHORIZED, "X-Keep-Wire": wire})
        assert answer.status_code == 400 and fake.calls == []

    def test_a_caller_one_version_away_is_served_and_told_this_sides_version(self, satellite, fake):
        for theirs in (Wire.WIRE_VERSION - 1, Wire.WIRE_VERSION + 1):
            answer = post(satellite, "scheduled", {}, {**AUTHORIZED, "X-Keep-Wire": str(theirs)})
            assert answer.status_code == 200 and answer.headers[Wire.WIRE_HEADER] == str(Wire.WIRE_VERSION)
        assert fake.calls == [("scheduled",), ("scheduled",)]

    def test_a_caller_two_versions_away_is_refused_but_still_followed(self, satellite, fake, follower):
        """The server moved on: no call of its newer wire is run, and its version is the news to act on."""
        headers = {**AUTHORIZED, "X-Keep-Wire": str(Wire.WIRE_VERSION + 2), "X-Keep-Branch": "release", "X-Keep-Commit": "cd" * 20}
        answer = post(satellite, "scheduled", {}, headers)
        assert answer.status_code == 409 and fake.calls == []
        assert str(Wire.WIRE_VERSION + 2) in answer.json()["error"]
        assert follower.notices == [("release", "cd" * 20)]

    def test_a_call_this_side_does_not_have_is_a_404_that_names_it(self, satellite, fake):
        answer = post(satellite, "mark_read", {"chat": "any;+;chat2"})
        assert answer.status_code == 404 and "mark_read" in answer.json()["error"] and fake.calls == []

    def test_the_binary_call_has_no_json_route(self, satellite, fake):
        """An attachment as base64 in JSON is what the wire is built to avoid."""
        fake.files[40] = b"\xff\xd8"
        assert post(satellite, "attachment", {"attachment_id": 40}).status_code == 404

    def test_an_argument_this_side_does_not_know_is_left_out(self, satellite, fake):
        answer = post(satellite, "resolve", {"recipient": "Alice", "tone": "warm"})
        assert answer.status_code == 200 and fake.calls == [("resolve", "Alice", "")]


@pytest.mark.unit
class TestWhatABodyMayBe:
    def test_only_a_json_object_of_the_calls_types_reaches_the_host(self, satellite, fake):
        url = f"{satellite.address}/messages/poll"
        answers = [
            requests.post(url, data="after_rowid=0&limit=10", headers=AUTHORIZED, timeout=5),
            requests.post(url, data='{"after_rowid": 0', headers={**AUTHORIZED, "Content-Type": "application/json"}, timeout=5),
            requests.post(url, data='{"after_rowid": 0, "horizon_ts": null, "limit": 10}',
                          headers={**AUTHORIZED, "Content-Type": "text/plain"}, timeout=5),
            post(satellite, "poll", [0, None, 10]),
            post(satellite, "poll", {"after_rowid": "0 OR 1=1", "horizon_ts": None, "limit": 10}),
            post(satellite, "poll", {"after_rowid": 0, "horizon_ts": None}),
            requests.post(url, headers=AUTHORIZED, timeout=5),
        ]
        assert [a.status_code for a in answers] == [400] * 7
        assert all(a.headers["Content-Type"].startswith("application/json") and a.json()["error"] for a in answers)
        assert fake.calls == []

    def test_a_body_over_the_limit_is_refused_unread(self, satellite, fake):
        answer = post(satellite, "send", {"recipient": "Alice", "body": "x" * (Wire.MAX_REQUEST_BYTES + 1)})
        assert answer.status_code == 413 and fake.calls == []

    def test_a_text_just_under_the_limit_is_sent(self, satellite, fake):
        fake.known["alice"] = Recipient("+46701740605")
        body = "å" * 20_000
        assert post(satellite, "send", {"recipient": "Alice", "body": body}).status_code == 200
        assert fake.calls[0] == ("send", "Alice", body, "")

    def test_a_poll_reads_no_more_rows_than_the_satellite_allows(self, satellite, fake):
        post(satellite, "poll", {"after_rowid": 0, "horizon_ts": None, "limit": 10 ** 9})
        post(satellite, "poll", {"after_rowid": 0, "horizon_ts": None, "limit": -5})
        assert [call[3] for call in fake.calls] == [service_module.MAX_POLL_ROWS, 1]

    def test_an_attachment_is_asked_for_by_number_only(self, satellite, fake):
        fake.files[40] = b"\xff\xd8 bytes"
        get = lambda path: requests.get(f"{satellite.address}/messages/attachment/{path}", headers=AUTHORIZED, timeout=5)
        good = get("40")
        assert (good.status_code, good.content, good.headers["Content-Type"]) == (200, b"\xff\xd8 bytes", "application/octet-stream")
        assert [get(path).status_code for path in ("-1", "40.5", "..%2F..%2Fetc%2Fpasswd", "40/extra", "abc")] == [404] * 5
        assert requests.post(f"{satellite.address}/messages/attachment/40", headers=AUTHORIZED, timeout=5).status_code == 405

    def test_a_file_over_the_cap_is_refused_by_its_size_and_never_read(self, satellite, fake):
        """A video of gigabytes must not be loaded into the satellite's memory to learn that it
        is too large: were the file read, these few bytes would be the answer."""
        fake.files[40] = b"\xff\xd8 bytes"
        fake.sizes[40] = Wire.MAX_ATTACHMENT_BYTES + 1
        answer = requests.get(f"{satellite.address}/messages/attachment/40", headers=AUTHORIZED, timeout=5)
        assert answer.status_code == 413 and "40" in answer.json()["error"]
        fake.sizes[40] = Wire.MAX_ATTACHMENT_BYTES
        assert requests.get(f"{satellite.address}/messages/attachment/40", headers=AUTHORIZED, timeout=5).content == b"\xff\xd8 bytes"


@pytest.mark.unit
class TestTheServiceItself:
    def test_a_slow_call_does_not_hold_up_the_next_one(self, satellite):
        """The host's calls run off the event loop: a send waiting for Messages.app must not make
        the feed's poll, or a health check, wait with it."""
        import threading
        release, entered = threading.Event(), threading.Event()

        class Slow(type(satellite.service.host)):
            def send(self, recipient, body, owner=""):
                entered.set()
                release.wait(2)
                return super().send(recipient, body, owner)

        satellite.service.host = Slow()
        sender = threading.Thread(target=post, args=(satellite, "send", {"recipient": "Alice", "body": "hi"}))
        sender.start()
        try:
            assert entered.wait(2)
            assert post(satellite, "scheduled", {}).status_code == 200
        finally:
            release.set()
            sender.join(2)

    @pytest.mark.parametrize("handler, failure, status", [
        (service_module.SatelliteService._refused, NotFound(), 404),
        (service_module.SatelliteService._failed, OSError("disk full"), 500)], ids=["a refusal", "a failure"])
    def test_an_error_handler_answers_also_for_what_is_no_request(self, handler, failure, status):
        """Quart runs them for a WebSocket and for a failure before any request exists: `request` is not there."""
        body, code = asyncio.run(handler(failure))
        assert code == status and set(body) == {"error"}

    def test_an_idle_connection_outlives_the_wait_between_two_polls(self):
        """hypercorn's own 5 s is the feed's poll interval: each poll would race the close of the
        connection it reuses, and a lost race reads as a satellite that is away."""
        from assistant.integration.FeedRunner import FeedRunner
        from integrations.imessage.IMessageIntegration import POLL_INTERVAL
        keep_alive = service_module.hypercorn_config("127.0.0.1:0").keep_alive_timeout
        assert keep_alive >= 2 * max(POLL_INTERVAL, FeedRunner.AWAY_POLL_SECONDS)

    def test_without_a_follower_it_still_says_what_it_runs(self, satellite):
        satellite.service.follower = None
        status = post(satellite, "health", {}).json()["satellite"]
        assert status["wire"] == Wire.WIRE_VERSION and status["update_failure"] == "" and status["updating"] is False

    def test_the_routes_are_the_interfaces_calls(self, satellite):
        """Nothing is served that MessagesHost does not define."""
        rules = {(rule.rule, tuple(sorted(rule.methods - {"OPTIONS", "HEAD"})))
                 for rule in satellite.service.app.url_map.iter_rules()}
        assert rules == {("/messages/<name>", ("POST",)), ("/messages/attachment/<int:attachment_id>", ("GET",))}
        assert set(Wire.calls()) < set(MessagesHost.__abstractmethods__)
