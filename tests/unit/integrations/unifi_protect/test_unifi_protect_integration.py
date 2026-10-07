import base64
import io
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from integrations.unifi_protect.UnifiProtectIntegration import (
    UnifiProtectIntegration, UnifiProtectError, UnifiProtectConfigurationError, MAX_DIMENSION,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    UnifiProtectIntegration.reset_instance()
    yield
    UnifiProtectIntegration.reset_instance()


CONFIG = {"host": "192.0.2.10", "api_key": "key-abc"}

CAMERAS = [
    {"id": "c1", "name": "Front Door", "state": "CONNECTED"},
    {"id": "c2", "name": "Garden", "state": "CONNECTED"},
    {"id": "c3", "name": "Garage", "state": "DISCONNECTED"},
]


def jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "blue").save(buf, format="JPEG")
    return buf.getvalue()


def response(*, json=None, content=b"", status=200) -> Mock:
    resp = Mock()
    resp.json.return_value = json
    resp.content = content
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return resp


def provider(config=CONFIG, cameras=CAMERAS, snapshot=jpeg(640, 480)):
    """A provider whose session answers the camera list and every snapshot request."""
    cam = UnifiProtectIntegration(config=config)
    session = Mock()
    session.headers = dict(cam._session.headers)

    def get(url, **kwargs):
        if url.endswith("/cameras"):
            return response(json=cameras)
        return response(content=snapshot)

    session.get.side_effect = get
    cam._session = session
    return cam


class TestConfiguration:
    def test_missing_api_key_is_a_configuration_error(self):
        with pytest.raises(UnifiProtectConfigurationError):
            UnifiProtectIntegration(config={"host": "192.0.2.10"})

    def test_a_missing_section_is_a_configuration_error(self):
        manager = Mock()
        manager.get.side_effect = KeyError("unifi_protect")
        with pytest.raises(UnifiProtectConfigurationError):
            UnifiProtectIntegration(config_manager=manager)

    def test_bare_host_gets_https_and_the_integration_api_path(self):
        cam = provider()
        cam.camera_names()
        url = cam._session.get.call_args.args[0]
        assert url == "https://192.0.2.10/proxy/protect/integration/v1/cameras"

    def test_host_with_scheme_and_trailing_slash_is_kept(self):
        cam = provider(config={**CONFIG, "host": "http://nvr.local/"})
        cam.camera_names()
        assert cam._session.get.call_args.args[0].startswith("http://nvr.local/proxy/")

    def test_api_key_header_contract_with_the_console(self):
        cam = UnifiProtectIntegration(config=CONFIG)
        assert cam._session.headers["X-API-KEY"] == "key-abc"

    def test_ssl_verification_is_off_unless_asked(self):
        assert UnifiProtectIntegration(config=CONFIG)._session.verify is False
        UnifiProtectIntegration.reset_instance()
        assert UnifiProtectIntegration(config={**CONFIG, "verify_ssl": True})._session.verify is True


class TestCameraNames:
    def test_lists_every_camera_the_console_has(self):
        assert provider().camera_names() == ["Front Door", "Garden", "Garage"]

    def test_ignore_keywords_drop_cameras(self):
        cam = provider(config={**CONFIG, "ignore_cameras": ["garage"]})
        assert cam.camera_names() == ["Front Door", "Garden"]

    def test_unreachable_console_gives_no_names_and_fails_health(self):
        cam = UnifiProtectIntegration(config=CONFIG)
        cam._session = Mock()
        cam._session.get.side_effect = ConnectionError("refused")
        assert cam.camera_names() == []
        assert cam.check_health() is False


class TestFetchSnapshots:
    def test_image_and_name_block_per_connected_camera(self):
        blocks = provider().fetch_snapshots(only=["front"])
        assert [b["type"] for b in blocks] == ["image", "text"]
        assert blocks[1]["text"] == "Camera: Front Door"
        decoded = base64.b64decode(blocks[0]["source"]["data"])
        assert Image.open(io.BytesIO(decoded)).size == (640, 480)

    def test_large_snapshot_is_resized_before_it_reaches_the_llm(self):
        blocks = provider(snapshot=jpeg(3840, 2160)).fetch_snapshots(only=["front"])
        decoded = base64.b64decode(blocks[0]["source"]["data"])
        width, height = Image.open(io.BytesIO(decoded)).size
        assert width == MAX_DIMENSION
        assert abs(width / height - 16 / 9) < 0.01

    def test_a_snapshot_that_cannot_be_resized_is_sent_as_it_came(self):
        original = jpeg(640, 480)
        with patch("integrations.unifi_protect.UnifiProtectIntegration.resize_to_jpeg", return_value=None):
            blocks = provider(snapshot=original).fetch_snapshots(only=["front"])
        assert base64.b64decode(blocks[0]["source"]["data"]) == original

    def test_a_camera_without_a_state_is_taken_as_connected(self):
        cam = provider(cameras=[{"id": "c9", "name": "Porch"}])
        blocks = cam.fetch_snapshots()
        assert [b["type"] for b in blocks] == ["image", "text"]
        assert cam._session.get.call_count == 2   # the list, then the snapshot

    def test_snapshot_asks_for_high_quality(self):
        cam = provider()
        cam.fetch_snapshots(only=["front"])
        snapshot_call = cam._session.get.call_args_list[-1]
        assert snapshot_call.args[0].endswith("/cameras/c1/snapshot")
        assert snapshot_call.kwargs["params"]["highQuality"] == "true"

    def test_offline_camera_is_reported_without_a_snapshot_request(self):
        cam = provider()
        blocks = cam.fetch_snapshots(only=["garage"])
        assert len(blocks) == 1 and blocks[0]["type"] == "text"
        assert "Garage" in blocks[0]["text"] and "offline" in blocks[0]["text"].lower()
        assert cam._session.get.call_count == 1   # only the list

    def test_failed_camera_does_not_stop_the_others(self):
        cam = provider()
        list_response = response(json=CAMERAS[:2])

        def get(url, **kwargs):
            if url.endswith("/cameras"):
                return list_response
            if "/c1/" in url:
                return response(status=500)
            return response(content=jpeg(320, 240))

        cam._session.get.side_effect = get
        blocks = cam.fetch_snapshots()
        assert [b["type"] for b in blocks] == ["text", "image", "text"]
        assert "Front Door" in blocks[0]["text"]
        assert blocks[2]["text"] == "Camera: Garden"

    def test_nothing_selected_fetches_nothing(self):
        cam = provider()
        assert cam.fetch_snapshots(only=["pool"]) == []
        assert cam._session.get.call_count == 1

    def test_unreachable_console_raises_for_the_command_to_log(self):
        cam = UnifiProtectIntegration(config=CONFIG)
        cam._session = Mock()
        cam._session.get.side_effect = ConnectionError("refused")
        with pytest.raises(UnifiProtectError):
            cam.fetch_snapshots()
