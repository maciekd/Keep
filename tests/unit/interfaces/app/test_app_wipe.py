"""`//wipe` and the app chat: the phones and the transcript are part of the state a wipe takes,
and the running server does not write them back before it stops."""
import base64

import pytest
from quart import Quart

from assistant import Backups, Destinations, Notifier
from common.Paths import CACHE_DIR
from interfaces.app import AppInterface as module
from interfaces.app import Devices as devices_module, Transcript as transcript_module
from interfaces.app.AppInterface import AppInterface
from interfaces.chat.ChatHub import ChatHub
from tests.unit.interfaces.chat.fakes import FakeAssistant

KEY = base64.b64encode(bytes(range(32))).decode()
IPHONE, TOKEN = "marcus-iphone-0001", "a1" * 32


@pytest.fixture
def started(tmp_path, monkeypatch):
    """The started interface with a registered phone, a message and a picture on disk."""
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(Notifier, "set_notifier", lambda hub: None)
    monkeypatch.setattr(Backups, "_wipe_listeners", [])
    ChatHub.reset_instance()
    Destinations.clear()
    interface = AppInterface()
    interface.start(FakeAssistant([]), Quart(__name__))
    transport = interface.transport
    transport.devices.register(IPHONE, TOKEN, "sandbox", KEY, "Marcus's iPhone")
    transport.transcript.upsert("note-1", "notice", text="The washing machine is done")
    transport.transcript.add_media(b"jpeg bytes", "jpg")
    transport.transcript.flush()
    yield interface
    interface.stop()
    ChatHub.reset_instance()
    Destinations.clear()


def files(tmp_path) -> set[str]:
    return {path.name for path in tmp_path.iterdir()}


def test_the_app_chats_files_are_wipe_targets_under_the_names_the_app_interface_writes():
    """Contract between `Backups.STATE` and `interfaces/app`, which the assistant cannot import."""
    state = [path for path, _ in Backups.STATE]
    for name in (devices_module.FILE_NAME, transcript_module.FILE_NAME, transcript_module.MEDIA_FOLDER):
        assert CACHE_DIR / name in state


def test_the_fixture_leaves_every_file_a_wipe_has_to_take(started, tmp_path):
    assert files(tmp_path) == {devices_module.FILE_NAME, transcript_module.FILE_NAME, transcript_module.MEDIA_FOLDER}


def test_after_a_wipe_the_stop_writes_nothing_of_the_old_user_back(started, tmp_path):
    Backups.wiped()
    assert files(tmp_path) == set()
    # what still happens before the stop: the wipe's own reply, a picture, a phone looking in
    started.transport.transcript.upsert("reply-1", "assistant", text="Wiped 12 item(s)", final=True)
    started.transport.transcript.add_media(b"png bytes", "png")
    started.transport.devices.register(IPHONE, TOKEN, "sandbox", KEY, "Marcus's iPhone")
    started.stop()
    assert files(tmp_path) == set()


def test_a_wiped_install_notifies_no_phone(started):
    Backups.wiped()
    assert started.transport.devices.all() == []
    assert started.transport.alerts() is False


def test_a_listening_app_still_reads_the_log_after_the_wipe(started):
    """The wipe's reply has to reach the app it was typed in."""
    Backups.wiped()
    assert [item["id"] for item in started.transport.transcript.items()] == ["note-1"]


def test_a_stopped_interface_is_no_longer_told_of_a_wipe(started):
    started.stop()
    assert Backups._wipe_listeners == []


def test_a_holder_that_fails_to_let_go_does_not_keep_the_others_from_it(monkeypatch):
    told = []
    monkeypatch.setattr(Backups, "_wipe_listeners", [])
    Backups.on_wipe(lambda: 1 / 0)
    Backups.on_wipe(lambda: told.append("second"))
    Backups.wiped()
    assert told == ["second"]
