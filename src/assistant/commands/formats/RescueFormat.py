"""The active command format with the other models' native blocks read beside it, so a call the model writes in its own trained shape under another syntax still runs.

Opus 5.5 writes a `<function_calls>` / `<invoke>` block now and then whatever syntax the
prompt teaches: four of 428 replies under Keep's syntax on the production server between
2026-10-18 and -21, each a whole, correct block and nothing else (Fable never did it once).
DeepSeek V4.1 Flash did the same with its DSML block on the ds4 host on 2026-11-07: five
replies in a row under Keep's syntax, each a DSML block, and not one call ran. Keep's parser
saw no call in them, so the raw block reached the chat, and the stored turn then showed the
model its own block on every later prompt — the loop that makes the drift recur. This format
wraps the active one: a buffer is read by whichever syntax opens a call first in it, the
active one or one of the rescues (`RESCUES`: Claude's invoke block, DeepSeek's DSML block),
a call a rescue read is logged at NOTE level and runs like any other, and `rewritten` writes
every such block back in the active syntax (its `render_tool_call`, the way `OllamaLLM`
writes a server-parsed call) before the turn is stored, so the history holds nothing but the
syntax the prompt teaches. A format that is itself one of the rescues keeps the others beside
it (the invoke format reads DSML as its rescue, the DSML format the invoke block); the fused spelling Opus wrote once (`<invoke_load_toolkit>`, 2026-10-22) and the DSML
slips of 2026-11-07 are each the rescue's own to read and are rescued the same way.
"""
from typing import Iterator, Sequence

from common.logging import info, note

from .CommandFormat import CommandFormat
from .DsmlFormat import DsmlFormat
from .InvokeFormat import InvokeFormat
from .Signatures import Declaration, Value


class RescueFormat(CommandFormat):
    """The active format, with the other native syntaxes read as a fallback for the calls the
    model writes in that shape instead; the texts render in the active format alone."""

    # The native block syntaxes a model slips into; each reads its own and offers `blocks`.
    RESCUES: tuple[type[CommandFormat], ...] = (InvokeFormat, DsmlFormat)

    def __init__(self, primary: CommandFormat, rescues: Sequence[CommandFormat]):
        super().__init__(primary.possible_commands)
        self.name = primary.name
        self.primary = primary
        self.rescues = list(rescues)
        self.rescued = 0  # the calls read by a rescue syntax so far (one processor, one reply)

    @classmethod
    def wrap(cls, fmt: CommandFormat) -> "RescueFormat":
        """`fmt` with the rescues it is not itself beside it."""
        return cls(fmt, [rescue(fmt.possible_commands) for rescue in cls.RESCUES if not isinstance(fmt, rescue)])

    # --- rendering: the active format's ------------------------------------------------------

    def instructions(self) -> str:
        return self.primary.instructions()

    def render_declaration(self, declaration: Declaration) -> str:
        return self.primary.render_declaration(declaration)

    def render_example(self, name: str, values: Sequence[Value], declaration: Declaration | None) -> str:
        return self.primary.render_example(name, values, declaration)

    def render_tool_call(self, name: str, arguments) -> str:
        return self.primary.render_tool_call(name, arguments)

    # --- the stream: whichever syntax opens a call first reads it -----------------------------

    @property
    def _readers(self) -> list[CommandFormat]:
        return [self.primary, *self.rescues]

    def _reader(self, buffer: str) -> CommandFormat:
        """The format whose call starts earliest in the buffer; the active one on a tie and
        when none opens a call."""
        reader, earliest = self.primary, self.primary.find_start(buffer)
        for rescue in self.rescues:
            start = rescue.find_start(buffer)
            if start >= 0 and (earliest < 0 or start < earliest):
                reader, earliest = rescue, start
        return reader

    def is_possibly_command(self, buffer: str, mid_line: bool = False) -> bool:
        return any(reader.is_possibly_command(buffer, mid_line) for reader in self._readers)

    def find_start(self, buffer: str, mid_line: bool = False) -> int:
        starts = [pos for reader in self._readers if (pos := reader.find_start(buffer, mid_line)) >= 0]
        return min(starts) if starts else -1

    def partial_tail(self, buffer: str, mid_line: bool = False) -> int:
        tails = [pos for reader in self._readers if (pos := reader.partial_tail(buffer, mid_line)) > 0]
        return min(tails) if tails else -1

    def complete(self, buffer: str, streaming: bool = False) -> bool:
        return self._reader(buffer).complete(buffer, streaming=streaming)

    def extract(self, buffer: str) -> tuple[tuple[str, list[str], str] | None, str | None]:
        reader = self._reader(buffer)
        call, rest = reader.extract(buffer)
        if reader is not self.primary and call is not None:
            self.rescued += 1
            note(f"Rescued a {call[0]} call the model wrote as a {reader.name} block under the {self.name} format "
                 f"— it runs as usual and is stored in {self.name} syntax (rescue #{self.rescued} of this reply)")
        return call, rest

    def text_after_quote(self, buffer: str, blocks_only: bool = False) -> str | None:
        return self._reader(buffer).text_after_quote(buffer, blocks_only)

    def call_end(self, buffer: str) -> int:
        return self._reader(buffer).call_end(buffer)

    def unterminated(self, buffer: str) -> str | None:
        return next((name for reader in self._readers if (name := reader.unterminated(buffer))), None)

    def _spans(self, text: str) -> Iterator[tuple[int, int, CommandFormat]]:
        """Every complete call block of the text in order, as `(start, end, the format that
        reads it)`; a call opened and never closed is stepped over."""
        at = 0
        while (start := self.find_start(text[at:])) >= 0:
            start += at
            reader = self._reader(text[start:])
            end = reader.call_end(text[start:])
            if end <= 0:
                at = start + 1
                continue
            yield start, start + end, reader
            at = start + end

    def extract_all(self, text: str) -> list[tuple[str, list[str]]]:
        return [call for start, end, reader in self._spans(text) for call in reader.extract_all(text[start:end])]

    def calls_in_prose(self, text: str) -> list[tuple[str, list[str]]]:
        return self.primary.calls_in_prose(text)

    def _cut_block(self, block: str, keep: int) -> str:
        return self._reader(block)._cut_block(block, keep)

    def rewritten(self, text: str) -> str:
        """The text with every block a rescue syntax read written in the active syntax, one
        call per line; a block of nothing (a closer nothing opened) is dropped."""
        out, at, blocks, calls = [], 0, 0, 0
        for start, end, reader in self._spans(text):
            if reader is self.primary:
                continue
            invokes = [invoke for _, _, block in reader.blocks(text[start:end]) for invoke in block]
            # A block runs wherever it stands; the call written in its place starts a line,
            # where the active syntax may read one (Keep's reads none inside a sentence) —
            # so do two blocks with nothing between them.
            out.append(text[at:start])
            if (written := "".join(out)) and not written.endswith("\n"):
                out.append("\n")
            out.append("\n".join(self.primary.render_tool_call(name, dict(arguments)) for name, arguments in invokes))
            at, blocks, calls = end, blocks + 1, calls + len(invokes)
        if not blocks:
            return text
        out.append(text[at:])
        info(f"Stored turn rewritten: {blocks} {'/'.join(r.name for r in self.rescues)} block(s) holding {calls} call(s) "
             f"written in {self.name} syntax")
        return "".join(out)
