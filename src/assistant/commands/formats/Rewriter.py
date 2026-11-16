"""Renders the command calls of a prompt text in the active command format, leaving everything else as written.

The texts (`res/prompt.md`, the toolkits) are written in Keep's syntax; `rewrite` finds every
backticked `name(...)` whose name is in the signature table and replaces it with the format's
rendering — `render_declaration` for a declaration, `render_example` for an example, whose
values are paired with the declared names in order. The texts follow one convention
(docs/plans/FORMAT_NEUTRAL_TEXTS_PLAN.md): a mention is the bare name, which stays as written;
`name()` is the declaration of a parameterless command and renders as one, never as a call
block. `name()` of a command that has parameters, a `word(x)` that is no command and a call the
canon cannot read are left as written rather than guessed at (a unit test forbids the first in
the texts). A passage between `<!-- keep -->` and `<!-- /keep -->` lines is prose only Keep's
syntax needs: the Keep format keeps the passage and drops the markers, every other format drops
the passage. For the Keep format the call rewrite is the identity.

The same gate reads two more pairs, for what depends on the system the server runs on
(common/Platform): a passage between `<!-- mac -->` and `<!-- /mac -->` lines stays on a Mac
alone, one between `<!-- linux -->` and `<!-- /linux -->` on a Linux server alone — so the Mac's
rendering of a text is what it was before the markers were put in. `gate_platform` is that part
on its own, for a text that never passes `rewrite` (the first-run brief).
"""
import re

from .CommandFormat import CommandFormat
from .Signatures import Declaration, calls, read_declaration, read_values
from common import Platform
from common.logging import trace

KEEP_ONLY_OPEN = "<!-- keep -->"
KEEP_ONLY_CLOSE = "<!-- /keep -->"
PLATFORM_MARKERS = (Platform.MAC, Platform.LINUX)


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- {name} -->", f"<!-- /{name} -->"


def _passages(name: str) -> re.Pattern:
    opening, closing = _markers(name)
    return re.compile(rf"^{re.escape(opening)}\n(.*?)^{re.escape(closing)}\n?", re.S | re.M)


_KEEP_ONLY_RE = _passages("keep")
_PLATFORM_RES = {name: _passages(name) for name in PLATFORM_MARKERS}


def rewrite(text: str, fmt: CommandFormat, table: dict[str, Declaration]) -> str:
    """`text` with its Keep-only passages kept or dropped for `fmt` and every declaration and
    example of a command in `table` rendered by it."""
    text = _gate(text, fmt)
    pieces, at = [], 0
    rendered, left = 0, []
    for call in calls(text):
        declaration = table.get(call.name)
        if declaration is None:
            continue
        own = read_declaration(call)
        if own is not None:
            replacement = fmt.render_declaration(own)
        elif not call.inner and not declaration.params:
            replacement = fmt.render_declaration(declaration)
        elif not call.inner:
            left.append(f"{call.name}() — a mention must be the bare name")
            continue
        else:
            values = read_values(call)
            if values is None or (len(values) > len(declaration.params) and not declaration.variadic):
                left.append(call.name)
                continue
            replacement = fmt.render_example(call.name, values, declaration)
        pieces.append(text[at:call.start])
        pieces.append(replacement)
        at = call.end
        rendered += 1
    if left:
        trace(f"Command rewriter ({fmt.name}): {len(left)} call(s) left as written, not canonical or "
              f"too many values: {', '.join(sorted(set(left)))}")
    if not rendered:
        return text
    pieces.append(text[at:])
    return "".join(pieces)


def _gate(text: str, fmt: CommandFormat) -> str:
    """The `<!-- keep -->` passages of `text` resolved for `fmt`. A marker without its pair is
    left as written: the text is not guessed at."""
    text = gate_platform(text)
    if KEEP_ONLY_OPEN not in text and KEEP_ONLY_CLOSE not in text:
        return text
    keep = fmt.name == "keep"
    gated, blocks = _KEEP_ONLY_RE.subn(lambda m: m.group(1) if keep else "", text)
    stray = gated.count(KEEP_ONLY_OPEN) + gated.count(KEEP_ONLY_CLOSE)
    trace(f"Command rewriter ({fmt.name}): {blocks} Keep-only passage(s) {'kept' if keep else 'dropped'}"
          + (f", {stray} unpaired marker(s) left as written" if stray else ""))
    return gated


def gate_platform(text: str) -> str:
    """The `<!-- mac -->` and `<!-- linux -->` passages of `text` resolved for the system the
    server runs on: its own are kept without their markers, the other system's are dropped. A
    marker without its pair is left as written."""
    system = Platform.current()
    for name, pattern in _PLATFORM_RES.items():
        if not any(marker in text for marker in _markers(name)):
            continue
        keep = name == system
        text, blocks = pattern.subn(lambda m: m.group(1) if keep else "", text)
        trace(f"Command rewriter: {blocks} {name}-only passage(s) {'kept' if keep else 'dropped'} on {system}")
    return text
