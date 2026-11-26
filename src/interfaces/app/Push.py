"""Notifications to the phones whose app is closed: the box only the phone opens, and the
sender that hands it to the relay (`docs/plans/APP_PUSH_PLAN.md`).

Apple delivers a notification only when the app's own key signed the request, and no install
holds that key: the relay at `RELAY_URL` does (`tools/push_relay/`). What it is handed is a
phone's token and a box — the notification's text, encrypted with the key that phone gave this
server when it registered. The relay and Apple learn that this server sent that phone something.

Nothing here waits for the network on a caller's thread, and nothing is sent twice: the text
is an item of the chat whether its notification arrives or not. No log line carries the text."""
import base64
import json
import os
import queue
import threading
from typing import Optional

import requests
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from assistant.Telemetry import Telemetry
from common.logging import info, trace
from interfaces.app.Devices import Device, Devices

RELAY_URL = "https://push.keep-ai.dev/push"
TIMEOUT_SECONDS = 15.0
QUEUE_SIZE = 50

NONCE_BYTES = 12
TAG_BYTES = 16
MAX_BOX_CHARS = 3000    # the relay's cap, under which Apple's 4 KB holds the whole notification
MAX_PLAIN_BYTES = MAX_BOX_CHARS // 4 * 3 - NONCE_BYTES - TAG_BYTES
BODY_BYTES = 2000       # of a notification's text; the chat has the rest
TITLE_CHARS = 80
CUT = "…"

# Apple's reasons for a token no notification will ever reach again.
DEAD_TOKEN = ("BadDeviceToken", "Unregistered", "DeviceTokenNotForTopic")


def cut(text: str, limit: int) -> str:
    """*text* in at most *limit* bytes of UTF-8, ended at a whole character."""
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    kept = data[:max(0, limit - len(CUT.encode("utf-8")))].decode("utf-8", errors="ignore")
    return kept.rstrip() + CUT


def plaintext(title: str, body: str, badge: int, kind: str) -> bytes:
    """What a phone's extension reads once the box is open — a contract with the app. The
    body is cut until the whole fits a box the relay takes."""
    title, body = title[:TITLE_CHARS], cut(body, BODY_BYTES)
    while True:
        data = json.dumps({"title": title, "body": body, "badge": badge, "kind": kind},
                          ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        over = len(data) - MAX_PLAIN_BYTES
        if over <= 0 or not body:
            return data
        # What JSON escapes (a quote, a line break) is longer in the box than in the text.
        body = cut(body, len(body.encode("utf-8")) - over - len(CUT.encode("utf-8")))


def seal(key: bytes, plain: bytes) -> str:
    """The box: a fresh nonce, the ciphertext and its tag, in that order, as base64 — what
    CryptoKit's `ChaChaPoly.SealedBox(combined:)` opens."""
    nonce = os.urandom(NONCE_BYTES)
    return base64.b64encode(nonce + ChaCha20Poly1305(key).encrypt(nonce, plain, None)).decode("ascii")


def post(body: dict) -> tuple[int, str]:
    """One notification handed to the relay: Apple's status and, when it refused, its reason."""
    response = requests.post(RELAY_URL, json=body, timeout=TIMEOUT_SECONDS)
    try:
        answer = response.json()
    except ValueError:
        answer = {}
    if response.status_code != 200:
        raise RuntimeError(f"the relay answered {response.status_code} {answer.get('error', '')}".rstrip())
    return int(answer.get("status") or 0), str(answer.get("reason") or "")


class Sender:
    """A queue and one thread: `push` returns at once, the notifications go out one after
    another."""

    def __init__(self, devices: Devices):
        self._devices = devices
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def push(self, device: Device, kind: str, body: str, title: str = "") -> None:
        try:
            self._queue.put_nowait((device, kind, title, body))
        except queue.Full:
            info(f"App push: {QUEUE_SIZE} notifications are waiting to go out — a {kind} for {device} is dropped")
            return
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="app-push", daemon=True)
                self._thread.start()

    def stop(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            trace("App push: the queue is full at stop, its thread ends with the process")

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                trace("App push: the sender stops")
                return
            device, kind = job[0], job[1]
            try:
                outcome = self.deliver(*job)
            except Exception as e:
                info(f"App push: a {kind} for {device} was not sent: {type(e).__name__}: {e}")
                outcome = "failed"
            Telemetry.get_instance().app_push(kind, outcome)

    def deliver(self, device: Device, kind: str, title: str, body: str) -> str:
        """One notification to one phone, on the caller's thread. The outcome: "sent", "gone"
        (the phone is no longer registered, or Apple no longer knows its token) or "refused"."""
        badge = self._devices.pushed(device.id)
        if not badge:
            trace(f"App push: {device} is no longer registered, its {kind} is not sent")
            return "gone"
        box = seal(device.key_bytes(), plaintext(title, body, badge, kind))
        status, reason = post({"token": device.token, "environment": device.environment, "box": box})
        if status == 200:
            info(f"App push: {kind} sent to {device} ({len(box)} chars, badge {badge})")
            return "sent"
        if status == 410 or reason in DEAD_TOKEN:
            self._devices.drop_token(device.token, reason or str(status))
            return "gone"
        info(f"App push: Apple refused a {kind} for {device}: {status} {reason}")
        return "refused"
