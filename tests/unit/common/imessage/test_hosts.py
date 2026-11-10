"""common/imessage/Hosts: the host a consumer's `host` and `host_token` keys name."""
from unittest.mock import patch

import pytest

from common.ConfigManager import config
from common.imessage import Hosts
from common.imessage.LocalMessagesHost import LocalMessagesHost
from common.imessage.MessagesHost import REFUSED
from common.imessage.RemoteMessagesHost import RemoteMessagesHost

from .satellite_fakes import TOKEN, server_identity


@pytest.fixture(autouse=True)
def no_local_host_kept():
    with patch.object(Hosts, "_local", None):
        yield


def configured(**values):
    """config.yaml with these keys, written `section__key`; every other key as the machine has it."""
    keys = {name.replace("__", "."): value for name, value in values.items()}
    real = config.get
    return patch.object(config, "get", side_effect=lambda key, default=None: keys[key] if key in keys else
                        default if key.split(".")[0] in ("imessage", "imessage_chat") else real(key, default))


@pytest.mark.unit
class TestLocal:
    @pytest.mark.parametrize("value", [None, "", "local", "Local", "  LOCAL "], ids=["absent", "empty", "local", "capital", "spaced"])
    def test_without_an_address_it_is_this_macs_own_messages(self, value):
        with configured(**({} if value is None else {"imessage__host": value})):
            assert not Hosts.is_remote("imessage")
            assert isinstance(Hosts.build("imessage"), LocalMessagesHost)

    def test_two_local_consumers_share_one_host(self):
        """One Messages account per macOS user: its send rate limit and contact cache are not per consumer."""
        with configured():
            first, second = Hosts.build("imessage"), Hosts.build("imessage_chat")
        assert first is second is Hosts.local()
        assert first.sender is second.sender


@pytest.mark.unit
class TestRemote:
    def test_an_address_is_a_satellite_reached_with_the_sections_token(self, satellite):
        with configured(imessage__host=satellite.address, imessage__host_token=TOKEN), server_identity():
            assert Hosts.is_remote("imessage")
            host = Hosts.build("imessage")
        assert isinstance(host, RemoteMessagesHost) and host.address == satellite.address
        assert host.health().ok, "the token of the section opened the satellite"

    def test_each_section_has_its_own_address_and_token(self, satellite):
        with configured(imessage__host=satellite.address, imessage__host_token="the-owners-satellite-token",
                        imessage_chat__host=f"{satellite.address}/", imessage_chat__host_token=TOKEN), server_identity():
            owner, chat = Hosts.build("imessage"), Hosts.build("imessage_chat")
        assert owner.health().reason == REFUSED and chat.health().ok

    def test_a_restarted_consumer_gets_a_host_built_from_the_config_as_it_is_now(self, satellite):
        with server_identity():
            with configured(imessage__host=satellite.address, imessage__host_token="an-old-token-typed-wrong"):
                before = Hosts.build("imessage")
            with configured(imessage__host=satellite.address, imessage__host_token=TOKEN):
                after = Hosts.build("imessage")
        assert before is not after and not before.health().ok and after.health().ok

    def test_an_address_without_a_token_is_a_host_that_says_so_not_an_exception(self):
        with configured(imessage__host="http://192.0.2.7:8898"), server_identity():
            host = Hosts.build("imessage")
        assert not host.health().ok and host.poll(5, None, 10).cursor == 5


@pytest.mark.unit
class TestSameAccount:
    """Two consumers on one Messages account would each get every message: the sections are compared."""

    def test_two_sections_left_local_are_one_account(self):
        with configured():
            assert Hosts.same_account("imessage", "imessage_chat")

    def test_local_and_a_satellite_are_two(self):
        with configured(imessage_chat__host="http://127.0.0.1:8898"):
            assert not Hosts.same_account("imessage", "imessage_chat")

    @pytest.mark.parametrize("one, other", [
        ("http://studio.local:8898", "http://studio.local:8898/"),
        ("http://Studio.local:8898", "http://studio.local:8898"),
        ("http://127.0.0.1:8898", "http://localhost:8898"),
    ], ids=["a trailing slash", "another case", "two names of this Mac"])
    def test_one_satellite_spelled_two_ways_is_one_account(self, one, other):
        with configured(imessage__host=one, imessage_chat__host=other):
            assert Hosts.same_account("imessage", "imessage_chat")

    @pytest.mark.parametrize("one, other", [
        ("http://127.0.0.1:8898", "http://127.0.0.1:8899"),
        ("http://studio.local:8898", "http://mini.local:8898"),
    ], ids=["two users of this Mac, two ports", "two Macs"])
    def test_two_satellites_are_two_accounts(self, one, other):
        with configured(imessage__host=one, imessage_chat__host=other):
            assert not Hosts.same_account("imessage", "imessage_chat")
