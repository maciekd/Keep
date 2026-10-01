"""HomeAssistantClient against a fake Home Assistant websocket server on localhost: the auth
handshake, message ids, result correlation, errors, subscriptions, drops and reconnects
(plans/HOME_ASSISTANT_PLAN.md)."""
import asyncio
import json
import threading
import time

import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from assistant.integration.integrations.home.HomeAssistantClient import HAError, HomeAssistantClient, NotConnected, websocket_url

TOKEN = "good-token"


class FakeHA:
    """Speaks the websocket protocol from a thread of its own: auth_required first, then
    answers per command from `results` ({type: result payload | {"error": {...}}}), records
    every message, pushes events to subscribers on demand."""

    def __init__(self, results=None, refuse=()):
        self.results = results or {}
        self.refuse = set(refuse)
        self.drop_on_subscribe = 0          # how many connections to close at their first subscribe, unanswered
        self.received: list[dict] = []
        self.connections = 0
        self._sockets = []
        self._subscriptions: dict = {}      # websocket → {event_type: id}
        self._loop = asyncio.new_event_loop()
        self._server = None
        self.port = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = threading.Event()
        self._thread.start()
        self._started.wait(5)

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        self._server = await serve(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        self._started.set()
        await self._server.wait_closed()

    async def _handle(self, ws):
        self.connections += 1
        self._sockets.append(ws)
        self._subscriptions[ws] = {}
        await ws.send(json.dumps({"type": "auth_required", "ha_version": "2026.9.1"}))
        auth = json.loads(await ws.recv())
        if auth.get("access_token") != TOKEN:
            await ws.send(json.dumps({"type": "auth_invalid", "message": "Invalid access token or password"}))
            await ws.close()
            return
        await ws.send(json.dumps({"type": "auth_ok", "ha_version": "2026.9.1"}))
        try:
            async for raw in ws:
                message = json.loads(raw)
                self.received.append(message)
                await self._answer(ws, message)
        finally:
            self._sockets.remove(ws)
            self._subscriptions.pop(ws, None)

    async def _answer(self, ws, message):
        kind, mid = message["type"], message["id"]
        if kind == "ping":
            await ws.send(json.dumps({"id": mid, "type": "pong"}))
        elif kind == "subscribe_events":
            if self.drop_on_subscribe:
                self.drop_on_subscribe -= 1
                await ws.close()
            elif message["event_type"] in self.refuse:
                await ws.send(json.dumps({"id": mid, "type": "result", "success": False,
                                          "error": {"code": "unauthorized", "message": "Unauthorized"}}))
            else:
                self._subscriptions[ws][message["event_type"]] = mid
                await ws.send(json.dumps({"id": mid, "type": "result", "success": True, "result": None}))
        elif kind == "slow":
            pass    # never answered
        elif kind in self.results:
            payload = self.results[kind]
            if isinstance(payload, dict) and "error" in payload:
                await ws.send(json.dumps({"id": mid, "type": "result", "success": False, "error": payload["error"]}))
            else:
                await ws.send(json.dumps({"id": mid, "type": "result", "success": True, "result": payload}))
        else:
            await ws.send(json.dumps({"id": mid, "type": "result", "success": False,
                                      "error": {"code": "unknown_command", "message": f"Unknown command {kind}"}}))

    def push(self, event_type: str, data: dict):
        async def _push():
            for ws, subs in list(self._subscriptions.items()):
                if event_type in subs:
                    try:
                        await ws.send(json.dumps({"id": subs[event_type], "type": "event",
                                                  "event": {"event_type": event_type, "data": data}}))
                    except ConnectionClosed:
                        pass    # a dropped socket whose handler has not wound down yet
        asyncio.run_coroutine_threadsafe(_push(), self._loop).result(5)

    def drop(self):
        async def _drop():
            for ws in list(self._sockets):
                await ws.close()
        asyncio.run_coroutine_threadsafe(_drop(), self._loop).result(5)

    def close(self):
        self.drop()
        self._loop.call_soon_threadsafe(self._server.close)
        self._thread.join(5)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"


def wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def server():
    fake = FakeHA(results={"get_states": [{"entity_id": "light.a", "state": "on"}],
                           "call_service": {"context": {"id": "x"}},
                           "config/auth/list": {"error": {"code": "unauthorized", "message": "Unauthorized"}}})
    yield fake
    fake.close()


@pytest.fixture
def client(server):
    c = HomeAssistantClient(server.url, TOKEN)
    c.reconnect_min = 0.01
    yield c
    c.stop()


@pytest.mark.unit
class TestUrl:
    def test_base_url_becomes_the_websocket_url(self):
        assert websocket_url("http://homeassistant.local:8123") == "ws://homeassistant.local:8123/api/websocket"
        assert websocket_url("https://ha.example.com/") == "wss://ha.example.com/api/websocket"
        assert websocket_url("192.168.1.10:8123") == "ws://192.168.1.10:8123/api/websocket"


@pytest.mark.unit
class TestHandshake:
    def test_connects_authenticates_and_reports_the_version(self, server, client):
        client.start()
        assert client.wait_ready(5)
        assert client.connected and client.ha_version == "2026.9.1"
        assert server.connections == 1

    def test_a_rejected_token_is_fatal_and_does_not_reconnect(self, server):
        client = HomeAssistantClient(server.url, "bad-token")
        client.reconnect_min = 0.01
        client.start()
        thread = client._thread
        assert wait_for(lambda: client.auth_failed)
        assert "token rejected" in client.last_error and "home_assistant.token" in client.last_error
        thread.join(5)     # the connection loop is over, so no reconnect delay is running
        assert not thread.is_alive()
        assert server.connections == 1
        with pytest.raises(NotConnected):
            client.call("ping")
        client.stop()

class NotHA(FakeHA):
    """Something else on the port (a reverse proxy, the wrong port): no auth_required."""

    async def _handle(self, ws):
        self.connections += 1
        await ws.send(json.dumps({"type": "hello"}))
        await ws.close()


class HalfHA(FakeHA):
    """Greets like Home Assistant, then answers the token with neither auth_ok nor auth_invalid."""

    async def _handle(self, ws):
        self.connections += 1
        await ws.send(json.dumps({"type": "auth_required"}))
        await ws.recv()
        await ws.send(json.dumps({"type": "surprise"}))
        await ws.close()


@pytest.mark.unit
class TestBrokenHandshake:
    @pytest.mark.parametrize("server_class, expected", [(NotHA, "expected auth_required"), (HalfHA, "expected auth_ok")])
    def test_a_server_that_is_not_home_assistant_is_retried_not_fatal(self, server_class, expected):
        """Only auth_invalid is final; anything else keeps reconnecting, since the user may be
        fixing the URL while the server runs."""
        server = server_class()
        client = HomeAssistantClient(server.url, TOKEN)
        client.reconnect_min = 0.05
        client.start()
        try:
            assert wait_for(lambda: server.connections >= 2)
            assert not client.auth_failed and not client.connected
            assert expected in client.last_error
        finally:
            client.stop()
            server.close()

    def test_a_failing_on_connect_hook_does_not_wedge_the_connection(self, server, client):
        """_refill is such a hook: a get_states that blows up must leave the socket usable."""
        client.on_connect(lambda: 1 / 0)
        client.start()
        assert client.wait_ready(5)
        assert client.call("ping") is None
        assert server.connections == 1


@pytest.mark.unit
class TestCalls:
    def test_ids_increase_and_results_are_correlated(self, server, client):
        client.start()
        assert client.wait_ready(5)
        assert client.call("ping") is None
        assert client.call("get_states") == [{"entity_id": "light.a", "state": "on"}]
        ids = [m["id"] for m in server.received]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)
        assert [m["type"] for m in server.received][-2:] == ["ping", "get_states"]

    def test_payload_goes_on_the_wire(self, server, client):
        client.start()
        assert client.wait_ready(5)
        client.call("call_service", domain="light", service="turn_on", target={"entity_id": "light.a"},
                    service_data={"brightness_pct": 40})
        sent = server.received[-1]
        assert sent["domain"] == "light" and sent["service"] == "turn_on"
        assert sent["target"] == {"entity_id": "light.a"} and sent["service_data"] == {"brightness_pct": 40}

    def test_a_failed_command_raises_with_the_servers_message(self, server, client):
        client.start()
        assert client.wait_ready(5)
        with pytest.raises(HAError) as e:
            client.call("config/auth/list")
        assert e.value.code == "unauthorized" and e.value.message == "Unauthorized"
        with pytest.raises(HAError) as e:
            client.call("no_such_command")
        assert "Unknown command" in e.value.message

    def test_a_slow_answer_times_out(self, server, client):
        client.start()
        assert client.wait_ready(5)
        with pytest.raises(TimeoutError):
            client.call("slow", timeout=0.05)
        assert client.call("ping") is None      # the connection is still fine

    def test_calls_from_several_threads_at_once(self, server, client):
        client.start()
        assert client.wait_ready(5)
        results = []
        threads = [threading.Thread(target=lambda: results.append(client.call("get_states"))) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        assert len(results) == 5 and all(r == [{"entity_id": "light.a", "state": "on"}] for r in results)


@pytest.mark.unit
class TestSubscriptions:
    def test_events_reach_the_handler_registered_before_start(self, server, client):
        seen = []
        client.subscribe("state_changed", seen.append)
        client.start()
        assert client.wait_ready(5)
        server.push("state_changed", {"entity_id": "light.a", "new_state": {"state": "off"}})
        assert wait_for(lambda: seen)
        assert seen[0]["event_type"] == "state_changed" and seen[0]["data"]["entity_id"] == "light.a"

    def test_a_refused_subscription_is_remembered_and_not_fatal(self, server, client):
        server.refuse.add("entity_registry_updated")
        client.subscribe("entity_registry_updated", lambda e: None)
        client.subscribe("state_changed", lambda e: None)
        client.start()
        assert client.wait_ready(5)
        assert client.refused("entity_registry_updated") and not client.refused("state_changed")

    def test_a_handler_that_raises_does_not_break_the_others(self, server, client):
        seen = []

        def bad(event):
            raise ValueError("boom")

        client.subscribe("state_changed", bad)
        client.subscribe("state_changed", seen.append)
        client.start()
        assert client.wait_ready(5)
        server.push("state_changed", {"entity_id": "light.a"})
        assert wait_for(lambda: seen)
        assert client.connected


@pytest.mark.unit
class TestHooksAndReconnect:
    def test_on_connect_hooks_run_before_ready_and_may_call(self, server, client):
        states = []
        client.on_connect(lambda: states.append(client.call("get_states")))
        client.start()
        assert client.wait_ready(5)
        assert states == [[{"entity_id": "light.a", "state": "on"}]]

    def test_a_drop_fails_pending_calls_reconnects_and_resubscribes(self, server, client):
        seen = []
        hooks = []
        client.subscribe("state_changed", seen.append)
        client.on_connect(lambda: hooks.append(time.time()))
        client.start()
        assert client.wait_ready(5)
        errors = []
        t = threading.Thread(target=lambda: errors.append(_catch(lambda: client.call("slow", timeout=5))))
        t.start()
        assert wait_for(lambda: any(m["type"] == "slow" for m in server.received))
        server.drop()
        t.join(5)
        assert isinstance(errors[0], NotConnected)
        assert client.disconnected_at is not None
        assert wait_for(lambda: server.connections == 2 and client.wait_ready(0.1))
        assert len(hooks) == 2
        server.push("state_changed", {"entity_id": "light.a"})
        assert wait_for(lambda: seen)

    def test_a_drop_during_the_subscribes_ends_the_session_and_reconnects(self, server, client):
        """A subscribe reply only ever comes through the reader: a socket dying between the
        request and its reply must not leave the session waiting for it forever."""
        server.drop_on_subscribe = 1
        client.subscribe("state_changed", lambda event: None)
        client.start()
        assert wait_for(lambda: server.connections == 2 and client.wait_ready(0.1))
        assert client.disconnected_at is not None
        assert client.call("ping") is None

    def test_stop_ends_the_thread(self, server, client):
        client.start()
        assert client.wait_ready(5)
        client.stop()
        assert not client.connected
        assert wait_for(lambda: not server._sockets)
        with pytest.raises(NotConnected):
            client.call("ping")

    def test_start_is_idempotent(self, server, client):
        client.start()
        assert client.wait_ready(5)
        client.start()
        time.sleep(0.05)
        assert server.connections == 1


def _catch(fn):
    try:
        return fn()
    except Exception as e:
        return e
