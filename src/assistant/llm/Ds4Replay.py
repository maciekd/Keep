"""Replays Keep's text history to ds4-server as the model wrote it, so a request extends the server's live KV context instead of prefilling it again.

ds4-server keeps one live KV context — the last prompt and the reply it sampled. With no tool
declared (Ds4LLM declares none) it renders past turns without their reasoning and keeps a
thinking checkpoint of that context: the next request continues it when its rendered text
extends the visible text the model wrote (`thinking live continuation match=visible-prefix` in
the server's log), and only the new tail is prefilled — a command round's result, ~0.3k–0.8k
tokens, where a full prefill costs ~7 s per 5k tokens on the test host. Keep's history strips
each assistant row (the thinking out, the surrounding whitespace trimmed), so a row sent as Keep
holds it rarely matches what the model sampled.

The store keeps, per reply, the verbatim content behind the row and the model's reasoning, keyed
by the stripped row so the conversion needs nothing from Keep but the rows. It is saved to one
small JSON file (`cache/ds4_replay.json`) after every reply, so a Keep restart costs no prefill;
it is bounded and the oldest entries go first. A row the store does not know goes out as the
text it is — the model still reads it, the cache misses once, and the next request is
consistent again.
"""
import json
import os
import re
from pathlib import Path

from common.logging import info, trace, note
from common.Paths import CACHE_DIR

DEFAULT_PATH = CACHE_DIR / "ds4_replay.json"
_THINKING_RE = re.compile(r"\A\s*<thinking>(.*?)</thinking>", re.S)


class ReplayStore:
    """The side store: assistant row (stripped) → its reasoning and verbatim content, bounded,
    mirrored to a JSON file when a path is given (None keeps it in memory — tests)."""

    LIMIT = 400

    def __init__(self, path: Path | None = DEFAULT_PATH, label: str = "ds4"):
        self.path = path
        self.label = label
        self.turns: dict[str, dict] = {}    # assistant row (stripped) → {"reasoning", "content"}
        self._loaded = path is None

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.turns = data.get("turns", {})
            info(f"{self.label} replay store loaded from {self.path}: {len(self.turns)} turn(s)")
        except FileNotFoundError:
            trace(f"{self.label} replay store: no {self.path} yet — starting empty")
        except (OSError, ValueError) as e:
            note(f"{self.label} replay store: {self.path} unreadable ({e}) — starting empty")

    def save(self):
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"turns": self.turns}, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            note(f"{self.label} replay store: could not write {self.path} ({e}) — kept in memory only")

    def remember_reply(self, reply: str, reasoning: str, content: str):
        """One finished reply: `reply` the text Keep stores as the assistant row (thinking
        left out, as yielded), `reasoning` the model's, `content` the reply as the model wrote
        it — what ds4 compares the next request's rendering with."""
        self._ensure_loaded()
        key = reply.strip()
        if not key or not (reasoning or content != key):
            trace(f"{self.label} replay store: a plain reply with nothing to remember")
            return
        self.turns.pop(key, None)
        self.turns[key] = {"reasoning": reasoning, "content": content}
        while len(self.turns) > self.LIMIT:
            del self.turns[next(iter(self.turns))]
        trace(f"{self.label} replay store: reply remembered ({len(reasoning)} reasoning chars)")
        self.save()

    def turn(self, key: str) -> dict | None:
        self._ensure_loaded()
        return self.turns.get(key)


def split_thinking(content: str) -> tuple[str | None, str]:
    """A row's leading `<thinking>…</thinking>` block (the reasoning as the stream wrapped it,
    still in the row within the request that produced it) and the rest."""
    match = _THINKING_RE.match(content)
    if not match:
        return None, content
    return match.group(1), content[match.end():]


def replay(messages: list[dict], store: ReplayStore) -> list[dict]:
    """The messages with every assistant row the store knows sent as the model wrote it, its
    reasoning as `reasoning_content`; a row the store does not know is passed as it is (its
    own thinking block, within the request, still goes as the reasoning). The stored content
    stands in only when it is the row's text: a reply whose call the server parsed out of it
    stores the text before the call, and the row then goes as Keep holds it."""
    out, replayed, reasoned = [], 0, 0
    for message in messages:
        if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
            out.append(message)
            continue
        reasoning_in_row, body = split_thinking(message["content"])
        turn = store.turn(body.strip())
        verbatim = turn is not None and turn["content"].strip() == body.strip()
        rebuilt = {"role": "assistant", "content": turn["content"] if verbatim else body}
        reasoning = turn["reasoning"] if turn else reasoning_in_row
        if reasoning:
            rebuilt["reasoning_content"] = reasoning
        out.append(rebuilt)
        replayed += verbatim
        reasoned += bool(reasoning)
    if replayed or reasoned:
        trace(f"{store.label} replay: {replayed} assistant row(s) sent as the model wrote them, {reasoned} with their reasoning")
    return out
