"""Finds the Hue bridges on this network: an mDNS query for `_hue._tcp.local` first (the
bridge answers it directly, with its address, whether or not it has internet), then Hue's
cloud discovery service, where the bridge registers its local address. The cloud service
rate-limits sharply (HTTP 429 after a handful of calls), so it is asked second and retried
once after the pause it names. SSDP is not tried: current bridge firmware no longer answers it."""
import socket
import struct
import time

import requests

from common.logging import info, trace

DISCOVERY_URL = "https://discovery.meethue.com/"
TIMEOUT = 10
RETRY_PAUSE = 2.0                 # the wait before the one retry when the 429 names no Retry-After
MAX_RETRY_PAUSE = 5.0             # longer than this and the user had better give the address
MDNS_GROUP = ("224.0.0.251", 5353)
MDNS_SERVICE = "_hue._tcp.local"
MDNS_WAIT = 2.0                   # how long to collect answers; the bridge answers within a few ms
_TYPE_PTR, _TYPE_A = 12, 1
_QU_CLASS = 0x8001                # class IN with the "unicast response" bit: answer me, not the group


def discover() -> list[str]:
    """The addresses of the bridges on this network, mDNS first and the cloud second; empty
    when neither knows one. Never raises: a failed lookup is logged and counts as empty."""
    hosts = mdns()
    if not hosts:
        hosts = cloud()
    return hosts


def mdns(wait: float = MDNS_WAIT) -> list[str]:
    """The bridges that answer an mDNS query for the Hue service: the IPv4 address each one
    puts in its answer, or failing that the address it answered from."""
    query = struct.pack("!HHHHHH", 0, 0, 1, 0, 0, 0) + _dns_name(MDNS_SERVICE) + struct.pack("!HH", _TYPE_PTR, _QU_CLASS)
    hosts: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
            sock.settimeout(wait)
            sock.sendto(query, MDNS_GROUP)
            deadline = time.monotonic() + wait
            while (remaining := deadline - time.monotonic()) > 0:
                sock.settimeout(remaining)
                try:
                    data, (sender, _) = sock.recvfrom(9000)
                except socket.timeout:
                    break
                for host in parse_answer(data) or [sender]:
                    if host not in hosts:
                        hosts.append(host)
    except OSError as e:
        info(f"Philips Hue: mDNS discovery failed: {e}")
        return []
    info(f"Philips Hue: mDNS found {len(hosts)} bridge(s)" + (f": {', '.join(hosts)}" if hosts else ""))
    return hosts


def cloud(sleep=time.sleep) -> list[str]:
    """The bridges Hue's discovery service lists for this network; retried once after a 429."""
    for attempt in (1, 2):
        try:
            response = requests.get(DISCOVERY_URL, timeout=TIMEOUT)
            if response.status_code == 429 and attempt == 1:
                pause = _retry_pause(response.headers.get("Retry-After"))
                info(f"Philips Hue: the discovery service is rate-limiting (429) — retrying once in {pause:.0f}s")
                sleep(pause)
                continue
            response.raise_for_status()
            hosts = [b["internalipaddress"] for b in response.json() if b.get("internalipaddress")]
        except (requests.RequestException, ValueError) as e:
            info(f"Philips Hue: cloud discovery failed: {e}")
            return []
        info(f"Philips Hue: cloud discovery found {len(hosts)} bridge(s)" + (f": {', '.join(hosts)}" if hosts else ""))
        return hosts
    return []


def _retry_pause(retry_after: str | None) -> float:
    try:
        return min(max(float(retry_after), 0.0), MAX_RETRY_PAUSE) if retry_after else RETRY_PAUSE
    except ValueError:
        return RETRY_PAUSE


def _dns_name(name: str) -> bytes:
    return b"".join(bytes([len(label)]) + label.encode() for label in name.strip(".").split(".")) + b"\x00"


def parse_answer(data: bytes) -> list[str]:
    """The IPv4 addresses of the A records in an mDNS answer (the bridge sends its own in the
    additional section); empty for a packet that holds none or cannot be read."""
    try:
        _, _, questions, answers, authorities, additionals = struct.unpack_from("!HHHHHH", data, 0)
        offset = 12
        for _ in range(questions):
            offset = _skip_name(data, offset) + 4
        hosts = []
        for _ in range(answers + authorities + additionals):
            offset = _skip_name(data, offset)
            rtype, _, _, length = struct.unpack_from("!HHIH", data, offset)
            offset += 10
            if rtype == _TYPE_A and length == 4:
                hosts.append(socket.inet_ntoa(data[offset:offset + 4]))
            offset += length
        return hosts
    except (struct.error, IndexError, OSError) as e:
        trace(f"Philips Hue: unreadable mDNS answer ({e})")
        return []


def _skip_name(data: bytes, offset: int) -> int:
    """The offset just past the name at *offset*; a compression pointer ends the name."""
    while True:
        length = data[offset]
        if length == 0:
            return offset + 1
        if length & 0xC0 == 0xC0:
            return offset + 2
        offset += 1 + length
