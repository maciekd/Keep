"""Reads a stored assistant reply back the way its stream showed it.

The history keeps a reply as the model wrote it — command calls and all, a `save_watch`
with its whole text among them — because that is what the LLM must see on its next turn.
A user watching the stream saw something else: the prose with the calls taken out, and
one echo per call ("Searching for …", `Commands.echo_text`). The reader rebuilds that
view from the stored text — and keeps, beside the echo, what a call said to somebody else
(`Commands.echo_body`, the text of a `post_message`), which no stream showed — with the parser the stream itself uses, so a chat redrawn from
the history reads like the one that was watched live.
"""
import re
from dataclasses import dataclass
from typing import NamedTuple

from common.logging import trace

from .formats.CommandFormats import CommandFormats
from .formats.KeepFormat import CODE_TAGS
from .Commands import Commands

WRAP_OPENS = re.compile(r"(`+|<code>)\s*$")
WRAP_CLOSES = re.compile(r"^\s*(`+|</code>)")
BLANK_RUN = re.compile(r"\n{3,}")


class Echo(NamedTuple):
    """One command call as a chat shows it."""
    text: str           # "Posting to Charlotte"
    body: str = ""      # what the call said there, when it said something


@dataclass
class StoredReply:
    """A stored reply as its stream showed it."""
    prose: str          # the reply without its command calls
    before: list[Echo]  # the echoes of the commands called before the prose began
    after: list[Echo]   # … and of those called once it had


class StoredReplyReader:
    """Splits a stored assistant reply into its prose and the echoes of its command calls."""

    def __init__(self):
        # Never runs a command: describing one needs no processor behind it.
        self._commands = Commands(None)
        self._format = CommandFormats.active()(self._commands.list_commands())

    def _follow_active_format(self):
        """The process's format is chosen when the smart LLM is built, which a reader made
        at startup precedes: a reply is read in the format it is written in now."""
        active = CommandFormats.active()
        if type(self._format) is not active:
            trace(f"StoredReplyReader: reading in the {active.name} command format now (was {self._format.name})")
            self._format = active(self._commands.list_commands())

    def read(self, reply: str) -> StoredReply:
        self._follow_active_format()
        prose, before, after = "", [], []
        remaining = reply or ""
        while remaining:
            start = self._format.find_start(remaining)
            if start < 0:
                break
            end = self._format.call_end(remaining[start:])
            if end <= 0:
                # A command name and "(" that never closes: the model was talking about it.
                prose += remaining[:start + 1]
                remaining = remaining[start + 1:]
                continue
            # The span as written, read whole: a block of several invokes is several calls.
            calls = self._format.extract_all(remaining[start:start + end])
            lead, remaining = self._unwrap(remaining[:start], remaining[start + end:])
            prose += lead
            for name, params in calls:
                echo = self._commands.echo_text(name, params)
                if echo:
                    body = self._commands.echo_body(name, params) or ""
                    (after if prose.strip() else before).append(Echo(echo, body))
                else:
                    trace(f"StoredReplyReader: {name} has no echo, nothing to show for it")
        prose = BLANK_RUN.sub("\n\n", prose + remaining).strip()
        return StoredReply(prose, before, after)

    @staticmethod
    def _unwrap(lead: str, rest: str) -> tuple[str, str]:
        """Drop the backticks or code tags a model wrapped a call in; without the call they
        would be left facing each other in the prose."""
        opening = WRAP_OPENS.search(lead)
        closing = WRAP_CLOSES.search(rest)
        if not opening or not closing:
            return lead, rest
        opens, closes = opening.group(1), closing.group(1)
        if (opens, closes) != CODE_TAGS and (opens != closes or not opens.startswith("`")):
            return lead, rest
        return lead[:opening.start()], rest[closing.end():]
