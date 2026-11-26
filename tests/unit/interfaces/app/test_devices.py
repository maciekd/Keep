"""The registry of the phones that take notifications: who is in it, the count behind a phone's
badge, and the file that holds their keys."""
import base64
import json

import pytest

from interfaces.app import Devices as devices_module
from interfaces.app.Devices import DeviceRefused, Devices

KEY = base64.b64encode(bytes(range(32))).decode()
OTHER_KEY = base64.b64encode(bytes(range(32, 64))).decode()
IPHONE, IPAD = "marcus-iphone-0001", "charlotte-ipad-0002"
TOKEN, NEW_TOKEN, IPAD_TOKEN = "a1" * 32, "b2" * 32, "c3" * 32


@pytest.fixture
def devices(tmp_path):
    return Devices(tmp_path)


def register(devices, device=IPHONE, token=TOKEN, environment="sandbox", key=KEY, name="Marcus's iPhone"):
    return devices.register(device, token, environment, key, name)


@pytest.mark.unit
class TestRegistering:
    def test_a_phone_seen_for_the_first_time_is_news_and_the_same_again_is_not(self, devices):
        assert register(devices) is True
        assert register(devices) is False
        assert [device.id for device in devices.all()] == [IPHONE]

    def test_a_phones_new_token_replaces_its_old_one(self, devices):
        register(devices)
        assert register(devices, token=NEW_TOKEN) is True
        assert [device.token for device in devices.all()] == [NEW_TOKEN]

    @pytest.mark.parametrize("change", [{"environment": "production"}, {"key": OTHER_KEY}],
                             ids=["the other of Apple's services", "a new key"])
    def test_a_phone_that_comes_back_changed_is_news(self, devices, change):
        register(devices)
        assert register(devices, **change) is True
        (device,) = devices.all()
        assert (device.environment, device.key) == (change.get("environment", "sandbox"), change.get("key", KEY))

    def test_a_token_written_in_capitals_is_the_same_address(self, devices):
        register(devices, token=TOKEN.upper())
        assert register(devices, token=TOKEN) is False

    def test_a_token_that_moved_to_another_id_leaves_one_phone(self, devices):
        """A reinstalled app makes up a new id; Apple's address of the phone is the same."""
        register(devices, device="marcus-iphone-old")
        register(devices, device=IPHONE)
        assert [device.id for device in devices.all()] == [IPHONE]

    def test_the_key_is_handed_back_as_the_bytes_the_phone_made(self, devices):
        register(devices)
        assert devices.get(IPHONE).key_bytes() == bytes(range(32))

    @pytest.mark.parametrize("field, value", [
        ("device", ""), ("device", "a phone / with a path"), ("device", None),
        ("token", "not hex at all"), ("token", "a1" * 8), ("token", ""),
        ("environment", "staging"), ("environment", None),
        ("key", base64.b64encode(b"too short").decode()), ("key", "not base64 !!"), ("key", None),
    ])
    def test_a_field_the_wire_does_not_know_is_refused_and_nothing_is_registered(self, devices, field, value):
        with pytest.raises(DeviceRefused):
            register(devices, **{field: value})
        assert devices.all() == []

    def test_a_refused_registration_leaves_the_phone_as_it_was(self, devices):
        register(devices)
        with pytest.raises(DeviceRefused):
            register(devices, token="gone wrong")
        assert devices.get(IPHONE).token == TOKEN


@pytest.mark.unit
class TestLeaving:
    def test_a_forgotten_phone_is_gone_and_cannot_be_forgotten_twice(self, devices):
        register(devices)
        register(devices, device=IPAD, token=IPAD_TOKEN)
        assert devices.forget(IPHONE) is True
        assert devices.forget(IPHONE) is False
        assert [device.id for device in devices.all()] == [IPAD]

    def test_a_token_apple_calls_dead_takes_its_phone_alone(self, devices):
        register(devices)
        register(devices, device=IPAD, token=IPAD_TOKEN)
        devices.drop_token(TOKEN, "Unregistered")
        assert [device.id for device in devices.all()] == [IPAD]

    def test_a_dead_token_of_no_phone_changes_nothing(self, devices):
        register(devices)
        devices.drop_token(NEW_TOKEN, "Unregistered")
        assert [device.id for device in devices.all()] == [IPHONE]

    def test_over_the_cap_the_phone_registered_longest_ago_leaves(self, devices, monkeypatch):
        monkeypatch.setattr(devices_module, "MAX_DEVICES", 3)
        for number in range(1, 5):
            register(devices, device=f"device-number-{number}", token=f"{number:02x}" * 32)
        assert [device.id for device in devices.all()] == ["device-number-2", "device-number-3", "device-number-4"]

    def test_a_phone_that_registered_again_is_not_the_one_to_leave(self, devices, monkeypatch):
        monkeypatch.setattr(devices_module, "MAX_DEVICES", 3)
        for number in range(1, 4):
            register(devices, device=f"device-number-{number}", token=f"{number:02x}" * 32)
        register(devices, device="device-number-1", token="01" * 32)
        register(devices, device="device-number-4", token="04" * 32)
        assert sorted(device.id for device in devices.all()) == ["device-number-1", "device-number-3", "device-number-4"]


@pytest.mark.unit
class TestTheBadge:
    def test_the_count_rises_with_every_notification(self, devices):
        register(devices)
        assert [devices.pushed(IPHONE), devices.pushed(IPHONE), devices.pushed(IPHONE)] == [1, 2, 3]

    def test_each_phone_counts_for_itself(self, devices):
        register(devices)
        register(devices, device=IPAD, token=IPAD_TOKEN)
        devices.pushed(IPHONE)
        devices.pushed(IPHONE)
        assert devices.pushed(IPAD) == 1

    def test_a_phone_that_registers_has_been_looked_at(self, devices):
        register(devices)
        devices.pushed(IPHONE)
        devices.pushed(IPHONE)
        register(devices)
        assert devices.get(IPHONE).unseen == 0
        assert devices.pushed(IPHONE) == 1

    def test_a_phone_that_is_gone_counts_nothing(self, devices):
        assert devices.pushed(IPHONE) == 0


@pytest.mark.unit
class TestTheFile:
    def test_what_was_registered_is_there_after_a_restart_with_its_key_and_its_count(self, devices, tmp_path):
        register(devices)
        devices.pushed(IPHONE)
        (device,) = Devices(tmp_path).all()
        assert (device.id, device.token, device.environment, device.key, device.name, device.unseen) == (
            IPHONE, TOKEN, "sandbox", KEY, "Marcus's iPhone", 1)

    def test_a_count_set_back_by_a_registration_is_set_back_on_disk_too(self, devices, tmp_path):
        register(devices)
        devices.pushed(IPHONE)
        register(devices)
        assert Devices(tmp_path).get(IPHONE).unseen == 0

    def test_the_file_holds_the_keys_and_is_the_owners_alone_to_read(self, devices, tmp_path):
        register(devices)
        path = tmp_path / devices_module.FILE_NAME
        assert path.stat().st_mode & 0o777 == 0o600
        assert KEY in path.read_text()

    def test_a_forgotten_phone_is_out_of_the_file(self, devices, tmp_path):
        register(devices)
        devices.forget(IPHONE)
        assert Devices(tmp_path).all() == []

    def test_a_file_that_cannot_be_read_starts_an_empty_registry_that_heals(self, tmp_path):
        (tmp_path / devices_module.FILE_NAME).write_text("{ not json")
        devices = Devices(tmp_path)
        assert devices.all() == []
        register(devices)
        assert json.loads((tmp_path / devices_module.FILE_NAME).read_text())["devices"][0]["id"] == IPHONE

    def test_a_folder_that_is_not_there_yet_is_made(self, tmp_path):
        devices = Devices(tmp_path / "cache")
        register(devices)
        assert Devices(tmp_path / "cache").get(IPHONE) is not None

    def test_a_registry_without_a_folder_keeps_nothing_on_disk(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        devices = Devices()
        register(devices)
        assert devices.get(IPHONE) is not None and list(tmp_path.iterdir()) == []
