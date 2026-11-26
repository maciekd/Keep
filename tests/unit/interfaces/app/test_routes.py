"""The /app routes on a real Quart app, over a transport on a real ChatHub."""
import asyncio
import io
import json
import logging
import threading

import pytest
from PIL import Image
from quart import Quart

from assistant.AssistantEvents import Done, RoundEnd, Token
from interfaces.app import routes
from interfaces.app.AppTransport import CONVERSATION, AppTransport
from interfaces.app.Transcript import Transcript
from interfaces.chat.ChatHub import ChatHub
from tests.unit.interfaces.chat.fakes import FakeAssistant, wait_until
from tests.unit.waiting import FAILURE_BOUND


@pytest.fixture(autouse=True)
def reset_singleton():
    ChatHub.reset_instance()
    yield
    ChatHub.reset_instance()


@pytest.fixture
def transport(tmp_path):
    return AppTransport(Transcript(tmp_path, autosave=False))


@pytest.fixture
def hub(transport):
    hub = ChatHub()
    hub.register(transport)
    transport.start(hub)
    yield hub
    hub.stop()


@pytest.fixture
def app(transport, hub):
    app = Quart(__name__)
    routes.register(app, transport)
    return app


def post(app, path, body):
    async def run():
        response = await app.test_client().post(path, json=body)
        return response.status_code, await response.get_json()
    return asyncio.run(run())


def post_form(app, path, fields, pictures):
    """A message as an app sends it with pictures: a form, an `image` part for each."""
    boundary = "keep-test-boundary"
    body = b""
    for name, value in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    for number, data in enumerate(pictures):
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{number}.jpg"\r\n'
                 f'Content-Type: image/jpeg\r\n\r\n').encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    async def run():
        response = await app.test_client().post(
            path, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        return response.status_code, await response.get_json()
    return asyncio.run(run())


def jpeg(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (40, 200, 40)).save(out, format="JPEG")
    return out.getvalue()


def read_events(app, path, until, act=None):
    """The lines of the event stream up to the first one *until* accepts; *act* runs once
    the stream is open."""
    async def run():
        lines, rest = [], b""
        async with app.test_client().request(path) as connection:
            await connection.send_complete()
            if act is not None:
                await asyncio.to_thread(act)
            while True:
                rest += await asyncio.wait_for(connection.receive(), FAILURE_BOUND)
                *whole, rest = rest.split(b"\n")
                lines += [json.loads(line) for line in whole]
                if any(until(line) for line in lines):
                    # The stream never ends by itself, and leaving the block waits for it.
                    await connection.disconnect()
                    return lines
    return asyncio.run(run())


class TestEvents:
    def test_a_fresh_app_is_sent_the_log_then_every_item(self, app, transport):
        transport.send_text(CONVERSATION, "The laundry is done.")
        lines = read_events(app, "/app/events", lambda line: "item" in line)
        assert lines[0] == {"log": transport.transcript.log}
        assert lines[1]["item"]["html"] == "The laundry is done."

    def test_what_happens_while_listening_arrives_on_the_open_stream(self, app, transport):
        lines = read_events(app, "/app/events", lambda line: "item" in line,
                            act=lambda: transport.send_text(CONVERSATION, "Dinner is ready."))
        assert lines[-1]["item"]["html"] == "Dinner is ready."

    def test_a_cursor_gets_only_what_it_missed(self, app, transport):
        transport.send_text(CONVERSATION, "one")
        transport.send_text(CONVERSATION, "two")
        log = transport.transcript.log
        lines = read_events(app, f"/app/events?after=1&log={log}", lambda line: "item" in line)
        assert [line["item"]["html"] for line in lines if "item" in line] == ["two"]

    def test_the_stream_closing_ends_the_listening(self, app, transport):
        read_events(app, "/app/events", lambda line: "log" in line)
        assert wait_until(lambda: transport.transcript.listeners() == 0)

    def test_a_stopping_server_ends_the_open_stream(self, app, transport):
        async def run():
            async with app.test_client().request("/app/events") as connection:
                await connection.send_complete()
                await asyncio.wait_for(connection.receive(), FAILURE_BOUND)
                transport.transcript.end_listeners()
                # Leaving the block waits for the response to end: bounded, it ends by itself.
            return True
        assert asyncio.run(asyncio.wait_for(run(), FAILURE_BOUND))
        assert transport.transcript.listeners() == 0

    def test_registering_tells_the_http_server_how_to_end_the_streams(self, app, transport):
        from interfaces.Interface import run_shutdown_hooks
        _, subscription = transport.transcript.subscribe(0, "", lambda: None)
        run_shutdown_hooks(app)
        assert subscription.ended

    def test_a_cursor_that_is_no_number_is_refused(self, app):
        async def run():
            return (await app.test_client().get("/app/events?after=soon")).status_code
        assert asyncio.run(run()) == 400

    def test_a_quiet_stream_is_kept_open_with_an_empty_line(self, app, monkeypatch):
        """A phone tells a dead connection from a quiet chat by these."""
        monkeypatch.setattr(routes, "HEARTBEAT_SECONDS", 0.01)
        lines = read_events(app, "/app/events", lambda line: line == {})
        assert lines[-2:] == [{"live": True}, {}]

    def test_a_change_after_a_quiet_spell_still_arrives(self, app, transport, monkeypatch):
        monkeypatch.setattr(routes, "HEARTBEAT_SECONDS", 0.01)
        seen = []

        def quiet_then_changed(line):
            seen.append(line)
            if line == {} and seen.count({}) == 1:
                transport.send_text(CONVERSATION, "The door is locked")
            return "item" in line
        lines = read_events(app, "/app/events", quiet_then_changed)
        assert lines[-1]["item"]["html"] == "The door is locked"

    def test_the_line_fields_are_a_contract_with_the_apps(self, app, transport):
        transport.transcript.upsert("bubble-1", "user", text="hi", images=[])
        lines = read_events(app, "/app/events", lambda line: "item" in line)
        assert set(lines[1]) == {"seq", "item"}
        assert set(lines[1]["item"]) == {"id", "kind", "at", "order", "text", "images"}


class TestSend:
    def test_a_sent_message_is_a_turn_under_the_apps_own_id(self, app, hub, transport):
        hub.assistant = FakeAssistant([Token("Hello"), RoundEnd("Hello"), Done("Hello")])
        assert post(app, "/app/send", {"id": "bubble-1", "text": "hi", "user": "Charlotte"})[0] == 200
        assert wait_until(lambda: hub.assistant.requests)
        assert hub.assistant.requests[0][0].user == "Charlotte"
        assert transport.transcript.item("bubble-1")["text"] == "hi"
        assert wait_until(lambda: not hub.session_for(transport, CONVERSATION).busy)

    def test_an_empty_message_is_refused(self, app):
        assert post(app, "/app/send", {"id": "bubble-1", "text": " "})[0] == 400

    def test_a_form_with_pictures_is_one_turn_carrying_them_a_contract_with_the_apps(self, app, hub, transport):
        hub.assistant = FakeAssistant([Token("A lawn"), RoundEnd("A lawn"), Done("A lawn")])
        fields = {"id": "bubble-1", "text": "what is this?", "user": "Charlotte"}
        assert post_form(app, "/app/send", fields, [jpeg(40, 30), jpeg(8, 8)])[0] == 200
        assert wait_until(lambda: hub.assistant.requests)
        ((req, query),) = hub.assistant.requests
        assert (req.user, query, len(hub.assistant.images[0])) == ("Charlotte", "what is this?", 2)
        sizes = [(image["width"], image["height"]) for image in transport.transcript.item("bubble-1")["images"]]
        assert sizes == [(40, 30), (8, 8)]
        assert wait_until(lambda: not hub.session_for(transport, CONVERSATION).busy)

    def test_a_form_with_more_pictures_than_a_message_carries_is_refused_with_the_reason(self, app, transport):
        status, answer = post_form(app, "/app/send", {"id": "bubble-1", "text": ""}, [jpeg(2, 2)] * 11)
        assert status == 400 and answer["error"]
        assert transport.transcript.items() == []

    def test_a_form_with_a_file_that_is_no_picture_is_refused_whole_and_nothing_of_it_is_kept(self, app, hub, transport, tmp_path):
        hub.assistant = FakeAssistant([Done("unasked")])
        fields = {"id": "bubble-1", "text": "what is this?"}
        status, answer = post_form(app, "/app/send", fields, [jpeg(40, 30), b"%PDF-1.7 not a picture"])
        assert status == 400 and answer["error"]
        assert transport.transcript.items() == [] and hub.assistant.requests == []
        kept = [path for path in tmp_path.rglob("*") if path.is_file() and path.suffix in (".jpg", ".png")]
        assert kept == [], "the picture beside the unreadable one must not stay on disk without a bubble"

    def test_a_refused_message_can_be_sent_again_under_the_same_id(self, app, hub, transport):
        """The resend rule tells a message that arrived by its bubble: a refused one left none."""
        hub.assistant = FakeAssistant([Token("A lawn"), RoundEnd("A lawn"), Done("A lawn")])
        fields = {"id": "bubble-1", "text": "what is this?"}
        assert post_form(app, "/app/send", fields, [b"not a picture"])[0] == 400
        assert post_form(app, "/app/send", fields, [jpeg(40, 30)])[0] == 200
        assert wait_until(lambda: hub.assistant.requests)
        assert len(hub.assistant.images[0]) == 1
        assert wait_until(lambda: not hub.session_for(transport, CONVERSATION).busy)

    def test_a_message_with_every_picture_it_may_carry_fits_the_upload_limit(self, app):
        from interfaces.app.AppTransport import MAX_PICTURES
        a_phone_photo = 12 * 1024 * 1024
        assert app.config["MAX_CONTENT_LENGTH"] >= MAX_PICTURES * a_phone_photo

    def test_registering_never_lowers_a_limit_another_route_set(self, transport, hub):
        app = Quart(__name__)
        app.config["MAX_CONTENT_LENGTH"] = 10 * routes.MAX_REQUEST_BYTES
        routes.register(app, transport)
        assert app.config["MAX_CONTENT_LENGTH"] == 10 * routes.MAX_REQUEST_BYTES

    def test_a_form_over_the_limit_is_refused_as_json_and_is_no_turn(self, app, transport):
        app.config["MAX_CONTENT_LENGTH"] = 2000
        status, body = post_form(app, "/app/send", {"id": "m1", "text": "Look"}, [b"x" * 5000])
        assert status == 413 and body["error"]
        assert transport.transcript.item("m1") is None

    def test_a_json_message_over_the_limit_is_refused_as_json(self, app, transport):
        app.config["MAX_CONTENT_LENGTH"] = 2000
        status, body = post(app, "/app/send", {"id": "m1", "text": "x" * 5000})
        assert status == 413 and body["error"]

    def test_a_form_without_text_or_pictures_is_refused(self, app):
        assert post_form(app, "/app/send", {"id": "bubble-1", "text": ""}, [])[0] == 400

    def test_stop_cancels_the_reply_in_flight(self, app, hub, transport):
        gate = threading.Event()
        hub.assistant = FakeAssistant([Token("Well"), gate, Done("Well")])
        post(app, "/app/send", {"id": "bubble-1", "text": "go on"})
        assert wait_until(lambda: hub.assistant.stream_obj is not None)
        assert post(app, "/app/stop", {})[0] == 200
        assert wait_until(lambda: hub.assistant.stream_obj.killed)


class TestApproval:
    def test_an_approval_is_answered_over_the_route(self, app, hub, transport):
        result = {}
        thread = threading.Thread(target=lambda: result.update(
            d=hub.request_approval("Send the email?", timeout=FAILURE_BOUND)), daemon=True)
        thread.start()
        assert wait_until(lambda: transport.transcript.items())
        token = transport.transcript.items()[0]["token"]
        assert post(app, "/app/approval", {"token": token, "decision": "deny"})[0] == 200
        thread.join(timeout=FAILURE_BOUND)
        assert result["d"] == "deny"

    def test_an_answer_to_a_closed_question_is_gone(self, app):
        assert post(app, "/app/approval", {"token": "stale", "decision": "approve"})[0] == 410

    def test_a_decision_that_is_none_is_refused(self, app):
        assert post(app, "/app/approval", {"token": "t", "decision": "maybe"})[0] == 400


class TestSecret:
    def test_the_value_is_in_no_item_and_no_log_line(self, app, hub, transport, caplog):
        seen, result = [], {}
        thread = threading.Thread(target=lambda: result.update(
            d=hub.request_secret("Card number", "", lambda value: seen.append(value) or "Stored",
                                 timeout=FAILURE_BOUND)), daemon=True)
        thread.start()
        assert wait_until(lambda: transport.transcript.items())
        token = transport.transcript.items()[0]["token"]
        with caplog.at_level(logging.DEBUG):
            assert post(app, "/app/secret", {"token": token, "value": "4111 1111 1111 1111"})[0] == 200
            thread.join(timeout=FAILURE_BOUND)
        assert result["d"] == "accepted" and seen == ["4111 1111 1111 1111"]
        assert "4111" not in json.dumps(transport.transcript.items())
        assert "4111" not in caplog.text

    def test_a_null_value_cancels(self, app, hub, transport):
        result = {}
        thread = threading.Thread(target=lambda: result.update(
            d=hub.request_secret("Password", "", lambda value: "ok", timeout=FAILURE_BOUND)), daemon=True)
        thread.start()
        assert wait_until(lambda: transport.transcript.items())
        token = transport.transcript.items()[0]["token"]
        assert post(app, "/app/secret", {"token": token, "value": None})[0] == 200
        thread.join(timeout=FAILURE_BOUND)
        assert result["d"] == "cancel"

    def test_a_value_that_is_no_text_is_refused_and_the_question_stays_open(self, app, hub, transport):
        seen, result = [], {}
        thread = threading.Thread(target=lambda: result.update(
            d=hub.request_secret("Door code", "", lambda value: seen.append(value) or "Stored",
                                 timeout=FAILURE_BOUND)), daemon=True)
        thread.start()
        assert wait_until(lambda: transport.transcript.items())
        token = transport.transcript.items()[0]["token"]
        assert post(app, "/app/secret", {"token": token, "value": 4711})[0] == 400
        assert seen == [] and transport.transcript.items()[0]["outcome"] == ""
        assert post(app, "/app/secret", {"token": token, "value": "4711"})[0] == 200
        thread.join(timeout=FAILURE_BOUND)
        assert seen == ["4711"]

    def test_a_secret_nobody_asked_for_is_gone(self, app):
        assert post(app, "/app/secret", {"token": "stale", "value": "x"})[0] == 410


class TestMedia:
    def test_a_photo_is_served_by_the_address_its_item_names(self, app, transport):
        transport.transcript.upsert("p", "media", url=transport.transcript.add_media(b"jpeg-bytes", "jpg"), poster="")

        async def run():
            response = await app.test_client().get(transport.transcript.item("p")["url"])
            return response.status_code, response.mimetype, await response.get_data()
        assert asyncio.run(run()) == (200, "image/jpeg", b"jpeg-bytes")

    def test_a_video_player_gets_the_byte_range_it_asks_for_a_contract_with_the_apps(self, app, transport):
        data = bytes(range(64))
        url = transport.transcript.add_media(data, "mp4")

        async def run():
            response = await app.test_client().get(url, headers={"Range": "bytes=8-23"})
            return response.status_code, response.headers.get("Content-Range"), await response.get_data()
        assert asyncio.run(run()) == (206, "bytes 8-23/64", data[8:24])

    def test_an_unknown_id_is_not_found(self, app):
        async def run():
            return (await app.test_client().get("/app/media/" + "0" * 32 + ".jpg")).status_code
        assert asyncio.run(run()) == 404
