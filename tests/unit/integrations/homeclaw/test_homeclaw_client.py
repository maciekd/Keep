"""The socket client against a fake app: one JSON line each way, the envelope unwrapped, and
the app not running told apart from the app refusing."""
import os
import socket
import tempfile

import pytest

from integrations.homeclaw.HomeClawClient import HomeClawClient, HomeClawError, HomeClawUnavailable
from .conftest import FakeHomeClaw, ok


def test_the_data_of_the_envelope_is_returned_and_unset_args_are_not_sent():
    app = FakeHomeClaw({"get_accessory": lambda args: ok({"echo": args})})
    try:
        data = HomeClawClient(app.path).call("get_accessory", id="A1000001", refresh=None)
        assert data == {"echo": {"id": "A1000001"}}
        assert app.requests == [{"command": "get_accessory", "args": {"id": "A1000001"}}]
    finally:
        app.close()


def test_a_refusal_carries_the_apps_message():
    app = FakeHomeClaw({"control": {"success": False, "error": "Characteristic not found: power"}})
    try:
        with pytest.raises(HomeClawError, match="Characteristic not found") as caught:
            HomeClawClient(app.path).call("control", id="x", characteristic="power", value="true")
        assert not isinstance(caught.value, HomeClawUnavailable)
    finally:
        app.close()


def test_a_reply_that_is_not_json_is_an_error_not_a_crash():
    app = FakeHomeClaw({"status": b"<html>\n"})
    try:
        with pytest.raises(HomeClawError, match="not JSON"):
            HomeClawClient(app.path).call("status")
    finally:
        app.close()


def test_no_socket_file_means_the_app_is_not_running():
    with pytest.raises(HomeClawUnavailable, match="not running"):
        HomeClawClient("/tmp/no-such-dir-for-homeclaw/homeclaw.sock").call("status")


def test_a_stale_socket_nobody_listens_on_means_the_app_is_not_running():
    directory = tempfile.mkdtemp(prefix="hc", dir="/tmp")
    path = os.path.join(directory, "homeclaw.sock")
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(path)
    stale.close()           # the file stays, nothing accepts on it: what a crashed app leaves behind
    try:
        with pytest.raises(HomeClawUnavailable, match="not running"):
            HomeClawClient(path).call("status")
    finally:
        os.unlink(path)
        os.rmdir(directory)


def test_an_app_that_does_not_answer_in_time_is_an_error_not_the_app_missing():
    """The toolkit tells the LLM never to retry "not running"; a slow accessory behind a
    `control` with verify is a different thing and must not read as the app being gone."""
    import threading
    gate = threading.Event()
    app = FakeHomeClaw({"control": lambda args: (gate.wait(1), b"")[1]})
    try:
        with pytest.raises(HomeClawError, match="did not answer control") as caught:
            HomeClawClient(app.path, timeout=0.02).call("control", id="x", characteristic="power", value="true")
        assert not isinstance(caught.value, HomeClawUnavailable)
    finally:
        gate.set()
        app.close()
