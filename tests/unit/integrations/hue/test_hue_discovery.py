"""HueDiscovery: the mDNS answer parsed to the bridge's address, the query's socket handling,
the cloud service's 429 retried once, and the order the two are asked in."""
import socket
import struct
from unittest.mock import Mock, patch

import pytest
import requests

from integrations.hue import HueDiscovery
from integrations.hue.HueDiscovery import cloud, discover, mdns, parse_answer

MODULE = "integrations.hue.HueDiscovery"


def _label(text: str) -> bytes:
    return bytes([len(text)]) + text.encode()


def _name(text: str) -> bytes:
    return b"".join(_label(p) for p in text.split(".")) + b"\x00"


def _record(name: bytes, rtype: int, rdata: bytes) -> bytes:
    return name + struct.pack("!HHIH", rtype, 1, 10, len(rdata)) + rdata


def _answer(addresses=("192.0.2.10",), with_question=True) -> bytes:
    """What a bridge sends back to the PTR query: the PTR in the answer section, SRV, TXT and the
    A record(s) in the additional one, names compressed with pointers like the real packet."""
    service = _name("_hue._tcp.local")
    instance = _label("Hue Bridge - 1A2") + b"\xc0\x0c"                      # pointer back to the service name
    srv = struct.pack("!HHH", 0, 0, 443) + _label("001788ab1a2b") + b"\xc0\x16"  # host.local via a pointer
    txt = _label("bridgeid=001788fffeab1a2b") + _label("modelid=BSB002")
    records = [_record(b"\xc0\x0c", 12, instance), _record(instance, 33, srv), _record(instance, 16, txt)]
    records += [_record(_name("001788ab1a2b.local"), 1, socket.inet_aton(a)) for a in addresses]
    records.append(_record(_name("001788ab1a2b.local"), 28, b"\xfe\x80" + b"\x00" * 14))
    header = struct.pack("!HHHHHH", 0, 0x8400, 1 if with_question else 0, 1, 0, len(records) - 1)
    question = service + struct.pack("!HH", 12, 0x8001) if with_question else b""
    return header + question + b"".join(records)


@pytest.mark.unit
class TestParseAnswer:
    def test_the_a_record_in_the_additional_section_is_the_bridges_address(self):
        assert parse_answer(_answer()) == ["192.0.2.10"]

    def test_every_a_record_counts_and_aaaa_is_left_out(self):
        assert parse_answer(_answer(("192.0.2.10", "192.0.2.11"))) == ["192.0.2.10", "192.0.2.11"]

    def test_an_answer_without_a_question_section_parses_too(self):
        assert parse_answer(_answer(with_question=False)) == ["192.0.2.10"]

    @pytest.mark.parametrize("data", [b"", b"\x00" * 5, _answer()[:40]])
    def test_a_truncated_packet_is_empty_not_an_exception(self, data):
        assert parse_answer(data) == []


@pytest.fixture
def udp():
    """The query's socket faked: `replies` is what recvfrom hands back, then a timeout."""
    with patch(f"{MODULE}.socket.socket") as socket_cls:
        sock = socket_cls.return_value.__enter__.return_value
        sock.replies = []
        sock.recvfrom.side_effect = lambda size: sock.replies.pop(0) if sock.replies else (_ for _ in ()).throw(socket.timeout())
        yield sock


@pytest.mark.unit
class TestMdns:
    def test_asks_the_multicast_group_for_the_hue_service_with_a_unicast_reply(self, udp):
        mdns(wait=0.01)
        query, group = udp.sendto.call_args.args
        assert group == ("224.0.0.251", 5353)
        assert _name("_hue._tcp.local") in query
        assert query.endswith(struct.pack("!HH", 12, 0x8001))

    def test_the_addresses_in_the_answers_are_the_bridges_once_each(self, udp):
        udp.replies = [(_answer(), ("192.0.2.10", 5353)), (_answer(), ("192.0.2.10", 5353)),
                       (_answer(("192.0.2.11",)), ("192.0.2.11", 5353))]
        assert mdns(wait=0.01) == ["192.0.2.10", "192.0.2.11"]

    def test_an_answer_without_an_a_record_falls_back_to_the_sender(self, udp):
        udp.replies = [(b"\x00" * 12, ("192.0.2.20", 5353))]
        assert mdns(wait=0.01) == ["192.0.2.20"]

    def test_no_answer_within_the_wait_is_an_empty_list(self, udp):
        assert mdns(wait=0.01) == []

    def test_a_network_without_multicast_is_empty_not_an_exception(self, udp):
        udp.sendto.side_effect = OSError("No route to host")
        assert mdns(wait=0.01) == []


def _response(status, payload=None, headers=None):
    response = Mock(spec=requests.Response)
    response.status_code = status
    response.headers = headers or {}
    response.json.return_value = payload
    response.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}") if status >= 400 else None
    return response


@pytest.mark.unit
class TestCloud:
    def test_lists_the_addresses_the_service_knows(self):
        found = [{"id": "ecb5fa00", "internalipaddress": "192.0.2.10", "port": 443}, {"id": "x", "port": 443}]
        with patch(f"{MODULE}.requests.get", return_value=_response(200, found)):
            assert cloud(sleep=Mock()) == ["192.0.2.10"]

    def test_a_429_is_retried_once_after_the_pause_the_service_names(self):
        sleep = Mock()
        answers = [_response(429, headers={"Retry-After": "3"}), _response(200, [{"internalipaddress": "192.0.2.10"}])]
        with patch(f"{MODULE}.requests.get", side_effect=answers) as get:
            assert cloud(sleep=sleep) == ["192.0.2.10"]
        assert get.call_count == 2
        sleep.assert_called_once_with(3.0)

    @pytest.mark.parametrize("retry_after, pause", [(None, HueDiscovery.RETRY_PAUSE), ("soon", HueDiscovery.RETRY_PAUSE),
                                                    ("600", HueDiscovery.MAX_RETRY_PAUSE)])
    def test_a_missing_odd_or_huge_retry_after_is_capped_to_a_short_pause(self, retry_after, pause):
        sleep = Mock()
        headers = {"Retry-After": retry_after} if retry_after else {}
        with patch(f"{MODULE}.requests.get", side_effect=[_response(429, headers=headers), _response(200, [])]):
            cloud(sleep=sleep)
        sleep.assert_called_once_with(pause)

    def test_a_second_429_gives_up_without_raising(self):
        with patch(f"{MODULE}.requests.get", side_effect=[_response(429), _response(429)]) as get:
            assert cloud(sleep=Mock()) == []
        assert get.call_count == 2

    @pytest.mark.parametrize("failure", [requests.ConnectionError("offline"), requests.Timeout()])
    def test_a_failed_call_is_empty_not_an_exception(self, failure):
        with patch(f"{MODULE}.requests.get", side_effect=failure):
            assert cloud(sleep=Mock()) == []


@pytest.mark.unit
class TestDiscover:
    def test_mdns_first_and_the_cloud_not_asked_when_it_answers(self):
        with patch(f"{MODULE}.mdns", return_value=["192.0.2.10"]), patch(f"{MODULE}.cloud") as cloud_:
            assert discover() == ["192.0.2.10"]
        cloud_.assert_not_called()

    def test_the_cloud_is_the_fallback_when_mdns_hears_nothing(self):
        with patch(f"{MODULE}.mdns", return_value=[]), patch(f"{MODULE}.cloud", return_value=["192.0.2.11"]):
            assert discover() == ["192.0.2.11"]
