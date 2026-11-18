"""Keep's own call syntax, `command(params)` at the start of a line of the reply, as `CommandParser` reads it.

The parser is wrapped method for method: the raw, free-text and keep-brackets exceptions are
quirks of this syntax and live in the parser, not in the processor.
"""
import json
from typing import Sequence

from assistant.commands.CommandParser import CommandParser
from common.logging import trace

from . import Signatures
from .CommandFormat import CommandFormat
from .NamedParameterFormat import split_variadic
from .Signatures import Declaration, Parameter, Value, VARIADIC

CODE_TAGS = ("<code>", "</code>")

# The format-specific part of the prompt's COMMAND SYSTEM section (res/prompt.md's
# {COMMAND_SYSTEM}): how a call is written in this syntax. The examples are in the canonical
# form, so the rewriter's rendering of this text is the identity.
INSTRUCTIONS = """\
The commands use python syntax.

Emit every command as a plain, literal function call at the start of a line in your reply: the command name
followed by its arguments in parentheses, strings in double quotes and numbers bare, for example
`turn_on("Kitchen lamp")` or `dim("Kitchen lamp", 40)`. That exact text is what gets executed. Separate several
arguments with commas; a quote inside a string is written `\\"`; a long or multi-line value goes in a triple-quoted
block, `\"\"\"…\"\"\"`.

NEVER express a command as JSON, a tool-call object, or a fenced code block. Do NOT wrap commands in ```json ... ```
or any other fence, and do NOT use structures like `{"tool": "...", "arguments": {...}}` or
`{"name": "...", "parameters": {...}}`. The parser only understands the plain `command("argument")` form — anything
else is treated as ordinary text, so the action silently never runs.

A call written inside a sentence does not run: the user reads it as text."""


def render_declaration(declaration: Declaration) -> str:
    """A declaration as the texts write it: `` `name(param: type, other: type = default, ...)` ``."""
    items = [_render_parameter(p) for p in declaration.params]
    if declaration.variadic:
        items.append(VARIADIC)
    return f"`{declaration.command}({', '.join(items)})`"


def _render_parameter(parameter: Parameter) -> str:
    text = f"{parameter.name}: {parameter.type}"
    return text if parameter.default is None else f"{text} = {parameter.default}"


def render_tool_call(name: str, arguments: dict) -> str:
    """A call parsed by a server — `name` with its named `arguments` — as Keep's positional
    call text, `name("value", 1)`: the values in the order the signature table declares the
    parameters (a declared one left out before a given one passed as `""`, undeclared ones
    after in the order given, a variadic last parameter spread into values — a list as it is,
    a string split at its commas or read as the JSON list it spells), a string
    quoted with `"` and `\\` escaped — one holding a newline as a triple-quoted block —, any
    other value bare as JSON. An opaque command (`browse_eval`, the free-text commands) takes
    its values verbatim, joined as its handler splits them."""
    declaration = Signatures.declarations().get(name)
    names = declaration.names if declaration else []
    if name in CommandParser.RAW_PARAM_COMMANDS | CommandParser.FREE_TEXT_COMMANDS:
        # The handler splits the payload itself (improve_integration: "name, request"), so
        # every value goes in, in the declared order, as the model would have written it.
        ordered = [arguments[key] for key in names if key in arguments]
        ordered += [value for key, value in arguments.items() if key not in names]
        payload = ", ".join(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for v in ordered)
        return f"{name}({payload})"
    present = [i for i, key in enumerate(names) if key in arguments]
    values = [arguments.get(key, "") for key in names[:present[-1] + 1]] if present else []
    if declaration and declaration.variadic and present and present[-1] == len(names) - 1:
        last = values[-1]
        values = values[:-1] + (last if isinstance(last, list) else split_variadic(str(last)))
    extra = [value for key, value in arguments.items() if key not in names]
    if extra:
        trace(f"keep format: {name} has {len(extra)} argument(s) outside its declaration, passed after the declared ones")
    return f"{name}({', '.join(_render_argument(v) for v in values + extra)})"


def _render_argument(value) -> str:
    if isinstance(value, str):
        if "\n" in value:
            return f'"""{value}"""'
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return json.dumps(value, ensure_ascii=False)


class KeepFormat(CommandFormat):
    """The `command(params)` syntax, read by `CommandParser`; the texts are rendered exactly as
    they are written, since Keep's syntax is the one they are written in."""

    name = "keep"

    def __init__(self, possible_commands: list[str]):
        super().__init__(possible_commands)
        self.parser = CommandParser(possible_commands)

    def instructions(self) -> str:
        return INSTRUCTIONS

    def render_declaration(self, declaration: Declaration) -> str:
        return render_declaration(declaration)

    def render_example(self, name: str, values: Sequence[Value], declaration: Declaration | None) -> str:
        return f"`{name}({', '.join(self._render_value(v) for v in values)})`"

    @staticmethod
    def _render_value(value: Value) -> str:
        if value.kind == "string":
            return f'"{value.text}"'
        if value.kind == "block":
            return f'"""{value.text}"""'
        return value.text

    @classmethod
    def render_tool_call(cls, name: str, arguments) -> str:
        return render_tool_call(name, cls.arguments_dict(arguments))

    def is_possibly_command(self, buffer: str, mid_line: bool = False) -> bool:
        return self.parser.is_possibly_command(buffer, mid_line)

    def find_start(self, buffer: str, mid_line: bool = False) -> int:
        return self.parser.find_command_start(buffer, mid_line)

    def partial_tail(self, buffer: str, mid_line: bool = False) -> int:
        return self.parser.find_partial_command_tail(buffer, mid_line)

    def complete(self, buffer: str, streaming: bool = False) -> bool:
        return self.parser.find_complete_command(buffer, streaming=streaming)

    def extract(self, buffer: str) -> tuple[tuple[str, list[str], str] | None, str | None]:
        call, rest = self.parser.extract_command_and_params(buffer)
        if call is None:
            return None, None
        name, params = call
        return (name, params, ", ".join(params)), rest

    def text_after_quote(self, buffer: str, blocks_only: bool = False) -> str | None:
        return self.parser.text_after_quote(buffer, blocks_only)

    def call_end(self, buffer: str) -> int:
        return self.parser.call_end(buffer)

    def unterminated(self, buffer: str) -> str | None:
        return self.parser.unterminated_command(buffer)

    def extract_all(self, text: str) -> list[tuple[str, list[str]]]:
        return self.parser.extract_all_commands(text)

    def calls_in_prose(self, text: str) -> list[tuple[str, list[str]]]:
        return self.parser.calls_in_prose(text)
