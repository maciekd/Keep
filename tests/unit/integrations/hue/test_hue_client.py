"""HueClient: the CLIP v2 reads and writes with their headers, the bridge's errors as
HueError, pairing, and the session that gets past the bridge's certificate."""
import ssl
from unittest.mock import Mock, patch

import pytest
import requests

from common import SystemTrust
from integrations.hue.HueClient import API_PATH, HueClient, HueError, LinkButtonNotPressed, bridge_session, pair

MODULE = "integrations.hue.HueClient"


def _response(payload, status=200):
    response = Mock(spec=requests.Response)
    response.status_code = status
    response.json.return_value = payload
    response.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}") if status >= 400 else None
    return response


@pytest.fixture
def session():
    with patch(f"{MODULE}.requests.Session") as session_cls:
        session = session_cls.return_value
        session.headers = {}
        yield session


@pytest.fixture
def client(session):
    return HueClient(" bridge.local ", "app-key"), session


@pytest.fixture
def post(session):
    return session.post


@pytest.mark.unit
class TestBridgeSession:
    def test_https_goes_through_an_unverified_plain_openssl_context(self):
        # truststore (SystemTrust.install) verifies the chain whatever verify=False says, and macOS
        # rejects the bridge's certificate outright — so the adapter must bring its own context.
        session = bridge_session()
        context = session.get_adapter("https://192.0.2.10/").poolmanager.connection_pool_kw["ssl_context"]
        assert session.verify is False
        assert context.verify_mode == ssl.CERT_NONE and context.check_hostname is False
        assert type(context).__mro__[1:] == SystemTrust._PlainOpenSSLContext.__mro__[1:]


@pytest.mark.unit
class TestResources:
    def test_get_sends_the_application_key_and_returns_the_data(self, client):
        hue, session = client
        session.get.return_value = _response({"errors": [], "data": [{"id": "l-1"}]})
        assert hue.get("light") == [{"id": "l-1"}]
        session.get.assert_called_once_with(f"https://bridge.local{API_PATH}/light", timeout=10)
        assert session.headers["hue-application-key"] == "app-key"
        assert session.verify is False

    def test_put_sends_the_body_to_the_resource(self, client):
        hue, session = client
        session.put.return_value = _response({"errors": [], "data": [{"rid": "l-1"}]})
        hue.put("light", "l-1", {"on": {"on": True}})
        session.put.assert_called_once_with(f"https://bridge.local{API_PATH}/light/l-1", json={"on": {"on": True}}, timeout=10)

    def test_the_bridges_errors_become_a_hue_error(self, client):
        hue, session = client
        session.put.return_value = _response({"errors": [{"description": "resource not found"}], "data": []}, status=404)
        with pytest.raises(HueError, match="resource not found"):
            hue.put("light", "gone", {})

    def test_a_refused_key_is_a_hue_error_too(self, client):
        hue, session = client
        session.get.return_value = _response({"errors": [{"description": "unauthorized connection"}]}, status=403)
        with pytest.raises(HueError, match="unauthorized"):
            hue.get("bridge")

    def test_a_failure_without_json_raises_http_error(self, client):
        hue, session = client
        response = _response(None, status=502)
        response.json.side_effect = ValueError("no json")
        session.get.return_value = response
        with pytest.raises(requests.HTTPError):
            hue.get("light")

    def test_a_success_without_json_is_the_bridges_fault_not_http(self, client):
        hue, session = client
        response = _response(None, status=200)
        response.json.side_effect = ValueError("no json")
        session.get.return_value = response
        with pytest.raises(HueError, match="200"):
            hue.get("light")


@pytest.mark.unit
class TestPairing:
    def test_pair_returns_the_key_and_asks_for_a_client_key(self, post):
        post.return_value = _response([{"success": {"username": "abc123", "clientkey": "k"}}])
        assert pair("192.0.2.10") == "abc123"
        body = post.call_args.kwargs["json"]
        assert body["generateclientkey"] is True and "#" in body["devicetype"]
        assert post.call_args.args[0] == "https://192.0.2.10/api"

    def test_a_single_answer_object_is_read_like_a_list(self, post):
        post.return_value = _response({"success": {"username": "key-xyz"}})
        assert pair("192.0.2.10") == "key-xyz"

    @pytest.mark.parametrize("answers", [[], [{"foo": 1}]])
    def test_an_answer_with_neither_key_nor_error_is_unexpected(self, post, answers):
        post.return_value = _response(answers)
        with pytest.raises(HueError, match="unexpected"):
            pair("192.0.2.10")

    def test_the_link_button_error_has_its_own_exception(self, post):
        post.return_value = _response([{"error": {"type": 101, "address": "", "description": "link button not pressed"}}])
        with pytest.raises(LinkButtonNotPressed):
            pair("192.0.2.10")

    def test_another_refusal_is_a_hue_error(self, post):
        post.return_value = _response([{"error": {"type": 7, "description": "invalid value"}}])
        with pytest.raises(HueError, match="invalid value"):
            pair("192.0.2.10")
