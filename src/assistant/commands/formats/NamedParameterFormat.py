"""The common ground of the call syntaxes that name their parameters: the JSON function schema a declaration renders to, and the signature table that puts the named values back in Keep's positional order.

Qwen's `<tool_call>` block and DeepSeek's DSML block both call a function by name with named
parameters, and both were trained on tools declared as JSON schemas. Keep's commands take
positional strings, so a format of this kind maps each parameter name to its position through
the signature table (`Signatures.table`, read from the declarations in the texts); a command
or a parameter the table does not know keeps the order the model wrote. The last parameter of
a variadic declaration (`load_toolkit(name: str, ...)`) is one array-typed parameter whose
value — a JSON list, or names separated by commas — is split back into the positional values.
"""
import json
import re
from typing import Sequence

from assistant.commands.CommandParser import CommandParser
from common.logging import trace

from .CommandFormat import CommandFormat
from .Signatures import Declaration, Parameter, Value

# The commands whose handler takes the whole payload as one string and splits it itself
# (`improve_integration(name, request)`): several named values join into that one payload.
_OPAQUE = CommandParser.RAW_PARAM_COMMANDS | CommandParser.FREE_TEXT_COMMANDS

_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


def argument_text(value) -> str:
    """A parsed argument value as the model would have written it: a string as is, a number
    bare, a boolean `true`/`false`, a list or object as JSON."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or value is None or isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def split_variadic(text: str) -> list[str]:
    """The values of a variadic parameter: a JSON list's items, else the comma-separated names."""
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            items = json.loads(stripped)
            if isinstance(items, list):
                return [argument_text(item) for item in items]
        except ValueError:
            pass
    return [item.strip() for item in stripped.split(",") if item.strip()]


class NamedParameterFormat(CommandFormat):
    """A format whose calls name their parameters inside one block between two sentinels:
    declarations render as JSON function schemas, the named values of a call are ordered
    through the signature table, and the stream is buffered from the opener to the closer."""

    # The sentinels of the block a call is written in; a subclass sets both, and the pattern
    # that matches one invoke of the block for the cut.
    OPEN: str
    CLOSE: str
    INVOKE_RE: "re.Pattern[str]"

    def __init__(self, possible_commands: list[str], table: dict[str, list[str]] | None = None,
                 variadic: set[str] | None = None):
        """`table` is `command → [parameter names]` and `variadic` the commands whose last
        parameter takes a list; both are read from the texts (`Signatures`) when not given."""
        super().__init__(possible_commands)
        self._table = table
        self._variadic = variadic if variadic is not None or table is None else set()
        # The parameter names of the call read last that its command does not declare.
        self.undeclared: list[str] = []

    def function_schema(self, declaration: Declaration) -> dict:
        """The `{"type": "function", "function": {...}}` schema of a declaration: the four
        python types map to JSON schema types, any other type text (`YYYY-MM-DD`,
        `open|close|0-100`) is a string described by that text, a default makes the parameter
        optional, the variadic last parameter is an array."""
        properties, required = {}, []
        last = len(declaration.params) - 1
        for i, parameter in enumerate(declaration.params):
            schema = self._schema(parameter.type)
            if declaration.variadic and i == last:
                schema = {"type": "array", "items": schema}
            if parameter.default is None:
                required.append(parameter.name)
            else:
                schema["default"] = self._default(parameter.default)
            properties[parameter.name] = schema
        parameters = {"type": "object"}
        if required:
            parameters["required"] = required
        parameters["properties"] = properties
        return {"type": "function", "function": {"name": declaration.command, "parameters": parameters}}

    def render_declaration(self, declaration: Declaration) -> str:
        return "`" + json.dumps(self.function_schema(declaration), ensure_ascii=False) + "`"

    @staticmethod
    def _schema(type_text: str) -> dict:
        if type_text in _JSON_TYPES:
            return {"type": _JSON_TYPES[type_text]}
        return {"type": "string", "description": type_text}

    @staticmethod
    def _default(text: str):
        try:
            return json.loads(text)
        except ValueError:
            return text

    # --- an example --------------------------------------------------------------------------

    def render_example(self, name: str, values: Sequence[Value], declaration: Declaration | None) -> str:
        """The block calling `name` with `values` paired to the declared names in order: a
        variadic tail folded into one JSON list under the last name, a value equal to its
        parameter's default left out (the texts write `""` to skip an optional parameter —
        Keep's positional way; here the parameter is simply not named), a value past the
        declared names as `argN`. The subclass spells the block (`_render_block`)."""
        names = declaration.names if declaration else []
        arguments: list[tuple[str, str, bool]] = []
        for i, value in enumerate(values):
            if declaration and declaration.variadic and i >= len(names) - 1:
                rest = [v.value for v in values[i:]]
                arguments.append((names[-1], json.dumps(rest, ensure_ascii=False), False))
                break
            if i < len(names) and self._is_default(value, declaration.params[i]):
                continue
            key = names[i] if i < len(names) else f"arg{i + 1}"
            if value.kind == "bare":
                arguments.append((key, value.text, False))
            else:
                arguments.append((key, value.value, True))
        return self._render_block(name, arguments)

    def _render_block(self, name: str, arguments: Sequence[tuple[str, str, bool]]) -> str:
        """One call block of this syntax: `arguments` are `(parameter, value text, is string)`
        triples — the flag matters to DSML alone."""
        raise NotImplementedError

    @classmethod
    def _is_default(cls, value: Value, parameter: Parameter) -> bool:
        if parameter.default is None:
            return False
        given = cls._default(value.text) if value.kind == "bare" else value.value
        return given == cls._default(parameter.default)

    # --- the stream: a block between the sentinels ------------------------------------------

    def is_possibly_command(self, buffer: str, mid_line: bool = False) -> bool:
        stripped = buffer.strip()
        return not stripped or stripped.startswith(self.OPEN) or self.OPEN.startswith(stripped)

    def find_start(self, buffer: str, mid_line: bool = False) -> int:
        return buffer.find(self.OPEN)

    def partial_tail(self, buffer: str, mid_line: bool = False) -> int:
        for length in range(min(len(self.OPEN) - 1, len(buffer)), 0, -1):
            if buffer.endswith(self.OPEN[:length]):
                pos = len(buffer) - length
                return pos if pos > 0 else -1
        return -1

    def complete(self, buffer: str, streaming: bool = False) -> bool:
        start = buffer.find(self.OPEN)
        return start >= 0 and buffer.find(self.CLOSE, start) >= 0

    def call_end(self, buffer: str) -> int:
        """The whole block, invokes and all: `extract` re-renders the further invokes into its
        rest, so the rest's length says nothing about where the block ends as written. A block
        without an invoke is a span too — read out to no call, so a reader drops it."""
        if self.extract(buffer)[1] is None:
            return -1
        return buffer.find(self.CLOSE, buffer.find(self.OPEN)) + len(self.CLOSE)

    def _cut_block(self, block: str, keep: int) -> str:
        """The block's first `keep` invokes, as written, under the closer again."""
        start = block.find(self.OPEN)
        if start < 0:
            return block
        body = start + len(self.OPEN)
        matches = list(self.INVOKE_RE.finditer(block, body))
        if len(matches) <= keep:
            return block
        return block[:matches[keep - 1].end()] + "\n" + self.CLOSE

    # --- the parameters ----------------------------------------------------------------------

    def _positional(self, name: str, arguments: list[tuple[str, str]]) -> list[str]:
        """The positional parameters of a call: in the declared order when the table knows
        the command and every parameter the model named, else in the model's order. A
        declared parameter left out before a given one is passed as an empty string; the
        trailing ones left out are dropped. A variadic last parameter is split into its values."""
        names = self._names_of(name)
        given = dict(arguments)
        self.undeclared = [key for key in given if key not in (names or ())]
        if names is None or any(key not in names for key in given):
            how = "an undeclared command" if names is None else "parameter(s) outside the declaration"
            trace(f"{self.name} format: {name} parameters taken in the model's order ({how}): {list(given)}")
            params = [value for _, value in arguments]
        else:
            present = [i for i, key in enumerate(names) if key in given]
            params = [given.get(key, "") for key in names[:present[-1] + 1]] if present else []
            if self._is_variadic(name) and params and len(params) == len(names):
                params = params[:-1] + split_variadic(params[-1])
            trace(f"{self.name} format: {name} parameters ordered by the signature table: {list(given)} → {len(params)} value(s)")
        if name in _OPAQUE and len(params) > 1:
            trace(f"{self.name} format: {name} is a free-text command, its {len(params)} values joined into one payload")
            params = [", ".join(params)]
        return params

    def _names_of(self, name: str) -> list[str] | None:
        if self._table is None:
            from . import Signatures
            self._table = Signatures.table()
        return self._table.get(name)

    def _is_variadic(self, name: str) -> bool:
        if self._variadic is None:
            from . import Signatures
            self._variadic = {command for command, d in Signatures.declarations().items() if d.variadic}
        return name in self._variadic
