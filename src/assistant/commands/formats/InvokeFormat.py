"""The `<function_calls>` / `<invoke>` call syntax Claude writes its tool calls in, read and rendered for Keep's commands.

A call is a block between `<function_calls>` and `</function_calls>` holding one
`<invoke name="…">` per call, each with `<parameter name="…">value</parameter>` elements: the
spelling Anthropic's models were trained to call tools in (the API turns it into `tool_use`
blocks only when a request declares tools; with none declared, the text comes through as
written — probed 2026-10-20 on Opus 5.5 and Haiku 5.5, streaming and not, see the plan). A
value is raw text, a string without quotes and a number bare; nothing is escaped, and one
leading and one trailing newline are stripped so a value may stand on its own line. The
parameters are named: `NamedParameterFormat` orders them through the signature table, and a
variadic last parameter takes a JSON list. A declaration renders as the `<function>{…}</function>`
line Anthropic's own harness declares a tool with — the name and a JSON-schema `parameters`
object, no description (the prose after the line is the description here) — and an example as
the block itself. Until 2026-10-21 a declaration kept Keep's `name(param: type)` line, and Opus
filled that shape in as a call now and then: a hundred lines of a fillable call shape leak into
the replies, a schema does not.
`render_call` is also what a backend that parses the model's calls itself (`OllamaLLM`,
`Ds4LLM`) writes a parsed call back into when this is the process's format. Under every other
format this one is the rescue (`RescueFormat`): a block the model writes in this shape anyway
is read by it, run, and stored in the active syntax.

The format is offered by name alone, `command_format: invoke` on a tier: Claude runs Keep's
syntax by default, whose prompt is the shorter (a declaration here is a JSON-schema line, about
twice the tokens of Keep's), and the rescue below covers the reply Opus writes in this shape
anyway. It was `ClaudeLLM`'s default for one day, 2026-10-21 to -22.

A bare `<invoke …>…</invoke>` outside a `<function_calls>` block is read as a block of one: on
the full prompt Opus 5.5 leaves the wrapper out about one reply in three (measured 2026-10-21,
the same query on the same cached prompt — the API hands the text back as written, the short
probe never showed it), and the call inside is complete and correct every time. The closer it still writes after the
last of them is a block of nothing and is dropped, wherever in the reply a closer nothing opened
stands.

A fused invoke, `<invoke_load_toolkit>…</invoke_load_toolkit>` — the command name joined to
the tag with `_` or `-` instead of standing in a `name` attribute, the `<parameter>` elements as
in the real block — is read as an invoke of that name. Opus wrote one on the production server
on 2026-10-22 under Keep's syntax with no `<invoke` anywhere in its prompt; neither syntax read
it, so the raw XML went to Telegram and into the stored turn. The closer is the fused tag again.
"""
import json
import re
from typing import Sequence

from common.logging import trace

from .NamedParameterFormat import NamedParameterFormat, argument_text
from .Signatures import Declaration

OPEN = "<function_calls>"
CLOSE = "</function_calls>"
INVOKE_OPEN = "<invoke"
INVOKE_CLOSE = "</invoke>"
PARAMETER_OPEN = "<parameter"
PARAMETER_CLOSE = "</parameter>"
_INVOKE_RE = re.compile(rf'{INVOKE_OPEN}\s+name="([^"]*)"\s*>(.*?){INVOKE_CLOSE}', re.S)
_PARAMETER_RE = re.compile(rf'{PARAMETER_OPEN}\s+name="([^"]*)"\s*>\n?(.*?)\n?{PARAMETER_CLOSE}', re.S)
_INVOKE_NAME_RE = re.compile(rf'{INVOKE_OPEN}\s+name="([^"]*)"')
_FUSED_RE = re.compile(rf'{INVOKE_OPEN}([_-])([A-Za-z][\w-]*)\s*>(.*?)</invoke\1\2\s*>', re.S)
_FUSED_NAME_RE = re.compile(rf'{INVOKE_OPEN}([_-])([A-Za-z][\w-]*)\s*>')
FUNCTION_OPEN = "<function>"
FUNCTION_CLOSE = "</function>"

# The shape Anthropic's own tool-use instruction teaches, without the `<functions>` listing that
# follows it there: Keep's functions are declared throughout the prompt and the toolkits, each
# as the `<function>` line that listing would hold. No backticked call of a real command stands
# here: the rewriter would render it.
INSTRUCTIONS = f"""\
In this environment you have access to a set of functions you can use to act for the user. You invoke a function by \
writing a block like the following as part of your reply:

{OPEN}
{INVOKE_OPEN} name="$FUNCTION_NAME">
{PARAMETER_OPEN} name="$PARAMETER_NAME">$PARAMETER_VALUE{PARAMETER_CLOSE}
...
{INVOKE_CLOSE}
{CLOSE}

A string or number value is written as is, without quotes. A parameter declared with a trailing `...` takes any \
number of values, written as one JSON array (`["email", "web"]`). A block may hold several invokes; they run in order.

Your functions are declared throughout this prompt and in every toolkit you load, each as a {FUNCTION_OPEN} line in \
JSONSchema format: its properties are the parameters you write in `<parameter name="…">`, `required` lists the ones \
you must give, a property with a `default` is optional, an array property takes its values as one JSON array, and a \
property's `description` says how its value is spelled. There is one set of functions: a function declared further \
down in this prompt and one declared in a toolkit you loaded are equally available to you and are invoked in exactly \
the same format. Never say that a capability is missing because a function was not listed here, and never load a \
toolkit to look for a function that is already declared in front of you.

A function runs the moment you write its {OPEN} block, so never write one you do not mean to run, and never put \
the block in a code fence or quote it — a fenced block is treated as text and the action silently never runs. To \
merely mention a function, write its name as a plain word; a name on its own never runs."""


def render_call(name: str, arguments: Sequence[tuple[str, str]]) -> str:
    """One `<function_calls>` block invoking `name` with the `(parameter, value text)` pairs in order."""
    lines = [OPEN, f'{INVOKE_OPEN} name="{name}">']
    lines += [f'{PARAMETER_OPEN} name="{key}">{text}{PARAMETER_CLOSE}' for key, text in arguments]
    lines += [INVOKE_CLOSE, CLOSE]
    return "\n".join(lines)


class InvokeFormat(NamedParameterFormat):
    """Anthropic's `<invoke>` syntax: a `<function_calls>` block of named-parameter invokes,
    mapped to Keep's positional parameters through the signature table."""

    name = "invoke"
    OPEN = OPEN
    CLOSE = CLOSE
    INVOKE_RE = _INVOKE_RE

    # --- rendering -------------------------------------------------------------------------

    def instructions(self) -> str:
        return INSTRUCTIONS

    def render_declaration(self, declaration: Declaration) -> str:
        """The `<function>{"name": …, "parameters": {…}}</function>` line of Anthropic's own tool
        listing: the schema `NamedParameterFormat` builds, without the OpenAI `function` wrapper."""
        schema = self.function_schema(declaration)["function"]
        return FUNCTION_OPEN + json.dumps(schema, ensure_ascii=False) + FUNCTION_CLOSE

    @classmethod
    def render_tool_call(cls, name: str, arguments) -> str:
        """A parsed call as the block the model would have written: each argument spelled by
        `argument_text` (a string as is, a number bare, a list as JSON)."""
        return render_call(name, [(key, argument_text(value)) for key, value in cls.arguments_dict(arguments).items()])

    def _render_block(self, name: str, arguments: Sequence[tuple[str, str, bool]]) -> str:
        return render_call(name, [(key, text) for key, text, _ in arguments])

    # --- parsing ---------------------------------------------------------------------------

    # A block is a `<function_calls>` element, or a bare `<invoke>` element standing outside one.
    # A `</function_calls>` nothing opened is a block of nothing: a model that leaves the opener
    # out still writes the closer after its last bare invoke, and it is dropped, never shown.
    _OPENERS = (OPEN, INVOKE_OPEN, CLOSE)

    def is_possibly_command(self, buffer: str, mid_line: bool = False) -> bool:
        stripped = buffer.strip()
        return not stripped or any(stripped.startswith(o) or o.startswith(stripped) for o in self._OPENERS)

    def find_start(self, buffer: str, mid_line: bool = False) -> int:
        return self._first_opener(buffer, 0)

    def _first_opener(self, buffer: str, at: int) -> int:
        starts = [pos for o in self._OPENERS if (pos := buffer.find(o, at)) >= 0]
        return min(starts) if starts else -1

    def partial_tail(self, buffer: str, mid_line: bool = False) -> int:
        tails = [pos for o in self._OPENERS if (pos := self._tail_of(buffer, o)) > 0]
        return min(tails) if tails else -1

    @staticmethod
    def _tail_of(buffer: str, opener: str) -> int:
        for length in range(min(len(opener) - 1, len(buffer)), 0, -1):
            if buffer.endswith(opener[:length]):
                return len(buffer) - length
        return -1

    def _span(self, buffer: str, at: int = 0) -> tuple[int, int, int, int] | None:
        """`(start, end, body start, body end)` of the first complete block from `at`: `end` is
        where the text after the block begins, the body is where its invokes stand — inside
        the `<function_calls>` element, or the whole bare invoke. None when no block is complete."""
        start = self._first_opener(buffer, at)
        if start < 0:
            return None
        if buffer.startswith(CLOSE, start):
            return start, start + len(CLOSE), start, start
        if buffer.startswith(OPEN, start):
            closer, body = CLOSE, start + len(OPEN)
        elif fused := _FUSED_NAME_RE.match(buffer, start):
            closer, body = f"</invoke{fused.group(1)}{fused.group(2)}>", start
        else:
            closer, body = INVOKE_CLOSE, start
        close = buffer.find(closer, body)
        if close < 0:
            return None
        end = close + len(closer)
        return start, end, body, (close if closer is CLOSE else end)

    def complete(self, buffer: str, streaming: bool = False) -> bool:
        return self._span(buffer) is not None

    def call_end(self, buffer: str) -> int:
        span = self._span(buffer)
        return span[1] if span else -1

    def extract(self, buffer: str) -> tuple[tuple[str, list[str], str] | None, str | None]:
        span = self._span(buffer)
        if span is None:
            return None, None
        start, end, body, body_end = span
        invokes = self._invokes(buffer[body:body_end])
        if not invokes:
            trace(f"invoke format: a block without an invoke (or a closer nothing opened), dropped: {buffer[start:end]!r}")
            return None, buffer[end:]
        name, arguments = invokes[0]
        params = self._positional(name, arguments)
        # The further invokes of the block go back as blocks of their own, so the processor
        # reads them one by one from the rest.
        rest = "\n".join(render_call(n, a) for n, a in invokes[1:])
        return (name, params, ", ".join(params)), rest + buffer[end:]

    def unterminated(self, buffer: str) -> str | None:
        stripped = buffer.strip()
        if not stripped.startswith(self._OPENERS):
            return None
        if match := _INVOKE_NAME_RE.search(stripped):
            return match.group(1).lower()
        if match := _FUSED_NAME_RE.search(stripped):
            return match.group(2).lower()
        return None

    def extract_all(self, text: str) -> list[tuple[str, list[str]]]:
        return [(name, self._positional(name, arguments))
                for _, _, invokes in self.blocks(text) for name, arguments in invokes]

    def blocks(self, text: str) -> list[tuple[int, int, list[tuple[str, list[tuple[str, str]]]]]]:
        """Every block of the text in order as `(start, end, invokes)` — the invokes as
        `(name, [(parameter, value)])` with the parameter names as the model wrote them, so a
        caller may render the call in another syntax; a block of nothing has none."""
        found = []
        at = 0
        while span := self._span(text, at):
            start, end, body, body_end = span
            found.append((start, end, self._invokes(text[body:body_end])))
            at = end
        return found

    @staticmethod
    def _invokes(block: str) -> list[tuple[str, list[tuple[str, str]]]]:
        """The `(name, [(parameter, value)])` of every invoke in a block, in order — the named
        ones and the fused ones (`<invoke_name>`) in the order they stand."""
        found = [(m.start(), m.group(1), m.group(2)) for m in _INVOKE_RE.finditer(block)]
        found += [(m.start(), m.group(2), m.group(3)) for m in _FUSED_RE.finditer(block)]
        return [(name.lower(), [(p.group(1), p.group(2)) for p in _PARAMETER_RE.finditer(body)])
                for _, name, body in sorted(found)]
