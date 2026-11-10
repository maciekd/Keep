"""Test doubles for the chat contract: a recording transport and a scripted assistant."""
import threading
import time
from typing import Optional

from assistant.Assistant import AssistantBusy
from interfaces.chat.ChatTransport import ChatTransport


class FakeDraft:
    def __init__(self, transport, conversation):
        self.transport = transport
        self.conversation = conversation
        self.calls = []

    def _log(self, *call):
        self.calls.append(call)
        self.transport.calls.append(("draft:" + call[0], self.conversation) + call[1:])

    def show_progress(self, progress):
        self._log("progress", progress)

    def update(self, html):
        self._log("update", html)
        return self.transport.update_ok

    def finalize(self, html):
        self._log("finalize", html)
        return self.transport.finalize_ok

    def discard(self):
        self._log("discard")

    @property
    def texts(self):
        return [c[1] for c in self.calls if c[0] in ("update", "finalize")]

    @property
    def progress(self):
        return [c[1] for c in self.calls if c[0] == "progress"]

    @property
    def finalized(self) -> Optional[str]:
        final = [c[1] for c in self.calls if c[0] == "finalize"]
        return final[-1] if final else None

    @property
    def discarded(self) -> bool:
        return any(c[0] == "discard" for c in self.calls)


class FakeTransport(ChatTransport):
    name = "Fake"
    source = "TextChat"
    max_message_length = 0

    def __init__(self, owners=(1,), name="Fake", source="TextChat", max_message_length=0):
        self.name = name
        self.source = source
        self.max_message_length = max_message_length
        self.owners = list(owners)
        self.notify = None  # a subset of owners the proactive pushes go to; None = every owner
        self.calls = []
        self.drafts = []
        self.update_ok = True
        self.finalize_ok = True     # False: the platform does not take a bubble's final text
        self.turned = []            # (conversation, message ids) of the guest turns turned away as busy
        self.accept = True
        self.prompt_ref = 101
        self.hub = None
        self.attended_now = True

    def start(self, hub):
        self.hub = hub

    def owner_conversations(self):
        return list(self.owners)

    def attended(self):
        return self.attended_now

    def notification_conversations(self):
        return list(self.notify) if self.notify is not None else list(self.owners)

    def send_text(self, conversation, html, *, silent=False, preview=True):
        self.calls.append(("text", conversation, html, silent, preview))
        return self.accept

    def send_photo(self, conversation, data, caption=""):
        self.calls.append(("photo", conversation, data, caption))
        return self.accept

    def send_photo_album(self, conversation, photos):
        self.calls.append(("album", conversation, photos))
        return self.accept

    def send_video(self, conversation, data, caption="", thumbnail=None):
        self.calls.append(("video", conversation, data, caption, thumbnail))
        return self.accept

    def begin_draft(self, conversation):
        draft = FakeDraft(self, conversation)
        self.drafts.append(draft)
        self.calls.append(("begin_draft", conversation))
        return draft

    def set_status_line(self, conversation, html):
        self.calls.append(("status", conversation, html))

    def turned_away(self, conversation, message_ids):
        self.turned.append((conversation, list(message_ids)))

    def reply_started(self, conversation):
        self.calls.append(("reply_started", conversation))

    def reply_finished(self, conversation):
        self.calls.append(("reply_finished", conversation))

    def send_approval_prompt(self, conversation, token, title, details):
        self.calls.append(("approval", conversation, token, title, details))
        return self.prompt_ref

    def close_approval_prompt(self, conversation, prompt_ref, outcome_text):
        self.calls.append(("close", conversation, prompt_ref, outcome_text))

    def send_secret_prompt(self, conversation, token, title, details):
        self.calls.append(("secret", conversation, token, title, details))
        return self.prompt_ref

    def close_secret_prompt(self, conversation, prompt_ref, outcome_text):
        self.calls.append(("close_secret", conversation, prompt_ref, outcome_text))

    # --- helpers ---------------------------------------------------------------------------

    @property
    def texts(self):
        return [c[2] for c in self.calls if c[0] in ("draft:update", "draft:finalize")]

    @property
    def sent_texts(self):
        return [c[2] for c in self.calls if c[0] == "text"]

    @property
    def finalized(self):
        return [c[2] for c in self.calls if c[0] == "draft:finalize"]

    def status_calls(self, conversation=None):
        return [c for c in self.calls if c[0] == "status" and (conversation is None or c[1] == conversation)]


class FakeStream:
    """Yields scripted events; an event may be a (delay_seconds, event) tuple, or a
    threading.Event the stream waits on until the test sets it (a kill ends the wait)."""

    def __init__(self, events):
        self._events = iter(events)
        self.killed = False

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.killed:
                raise StopIteration
            item = next(self._events)
            if isinstance(item, threading.Event):
                while not item.wait(0.005):
                    if self.killed:
                        raise StopIteration
                continue
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (int, float)):
                delay, item = item
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline:
                    if self.killed:
                        raise StopIteration
                    time.sleep(0.005)
            return item

    def kill(self):
        self.killed = True


class FakeAssistant:
    def __init__(self, events=(), busy=None, raise_exc=None):
        self.events = list(events)
        self.busy = busy
        self.raise_exc = raise_exc
        self.requests = []
        self.images = []        # one entry per request: the images it carried, or None
        self.stream_obj = None

    def stream(self, req, query=None, images=None, request_data=None):
        self.requests.append((req, query))
        self.images.append(getattr(req, 'images', None) or images)
        if self.raise_exc:
            raise self.raise_exc
        if self.busy:
            raise AssistantBusy(self.busy)
        self.stream_obj = FakeStream(self.events)
        return self.stream_obj


def wait_until(predicate, timeout=3.0, interval=0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
