"""Keeps what a `browse_act` call types into a page out of the log: a fill is shown as its length.

The vault masks only the values it holds, and a card's expiry and security code are too short
for it. Which field a value goes to is known only once the browser tool has looked at the page,
so every place that writes the call itself — the reply as the model wrote it, the command line,
the labels the browser tool names an action by — shows no typed value at all; the tool's own
`fill [n] = …` line shows the ones that are no card or credential field."""
import re

CALL = "browse_act("
MASK = "•••"
# A batch is split only where an index follows, so a ';' inside a typed value stays in it.
_BATCH_SPLIT_RE = re.compile(r';\s*(?=\d+\s*(?:=|;|$))')
_QUOTES = "\"'"


def split_batch(spec: str) -> list[str]:
    """The actions of a spec: `<index>` or `<index>=<value>`, ';' between them."""
    return [part.strip() for part in _BATCH_SPLIT_RE.split(spec) if part.strip()]


def spec_label(spec: str) -> str:
    """*spec* as a log line or a note may show it: every index, and of a typed value its length."""
    return "; ".join(_action_label(action) for action in split_batch(spec or ""))


def _action_label(action: str) -> str:
    index, typed, value = action.partition("=")
    if not typed:
        return action
    value = value.strip()
    if value.startswith(MASK):      # masked already: a line may pass here more than once
        return f"{index.strip()}={value}"
    return f"{index.strip()}={MASK} ({len(value)} chars)"


def mask_fills(text: str) -> str:
    """*text* with the typed values of every `browse_act` call in it masked. A call ends at the
    last ')' of its line: a value may hold one, and masking too much costs a few words of a log
    line where masking too little leaves a card's code in it."""
    if CALL not in text:
        return text
    return "\n".join(_mask_line(line) if CALL in line else line for line in text.split("\n"))


def _mask_line(line: str) -> str:
    head, _, tail = line.partition(CALL)
    end = tail.rfind(")")
    spec, rest = (tail[:end], tail[end:]) if end >= 0 else (tail, "")
    inner = spec.strip()
    quote = inner[0] if inner[:1] in _QUOTES and inner else ""
    if quote:
        inner = inner[1:-1] if inner.endswith(quote) and len(inner) > 1 else inner[1:]
    return f"{head}{CALL}{quote}{spec_label(inner)}{quote}{rest}"
