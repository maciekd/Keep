"""Keeps what a `browse_act` call types into a page out of the log: a fill is shown as its length.

The vault masks only the values it holds, and a card's expiry and security code are too short
for it. Which field a value goes to is known only once the browser tool has looked at the page,
so every place that writes the call itself — the reply as the model wrote it, the command line,
the labels the browser tool names an action by — shows no typed value at all; the tool's own
`fill [n] = …` line shows the ones that are no card or credential field.

The call is known in every syntax a command format writes it in: `browse_act(…)`, and the markup
of the qwen, dsml and invoke formats, where the spec is the body of a parameter block that may
run over several lines."""
import re

CALL = "browse_act("
MASK = "•••"
# A batch is split only where an index follows, so a ';' inside a typed value stays in it.
_BATCH_SPLIT_RE = re.compile(r';\s*(?=\d+\s*(?:=|;|$))')
_QUOTES = "\"'"
NAME = CALL.rstrip("(")
# The call named in markup: `<function=browse_act>` (qwen), `invoke name="browse_act"` (dsml, invoke).
_MARKUP_CALL_RE = re.compile(r'(?:<function=|invoke\s+name=["\']?)' + NAME + r'\b')
_MARKUP_CALL_END_RE = re.compile(r'</[^<>\n]*(?:function|invoke)\s*>')
# A parameter block: its opening tag, the body, and what ends it — its closing tag, or the end
# of a call cut off inside it.
_PARAMETER_RE = re.compile(r'(<[^<>/\n]*parameter[^<>\n]*>)(.*?)(</[^<>\n]*parameter\s*>|\Z)', re.DOTALL)


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
    if NAME not in text:
        return text
    if _MARKUP_CALL_RE.search(text):
        text = _mask_markup(text)
    return "\n".join(_mask_line(line) if CALL in line else line for line in text.split("\n"))


def _mask_markup(text: str) -> str:
    """*text* with the parameter bodies of every `browse_act` call written as markup masked: from
    the tag that names the call to the one that closes it — to the end of the text for a call
    that was cut off, where masking too much is again the safe side."""
    out, at = [], 0
    for call in _MARKUP_CALL_RE.finditer(text):
        if call.end() < at:
            continue
        closing = _MARKUP_CALL_END_RE.search(text, call.end())
        end = closing.start() if closing else len(text)
        out.append(text[at:call.end()])
        out.append(_PARAMETER_RE.sub(_masked_parameter, text[call.end():end]))
        at = end
    out.append(text[at:])
    return "".join(out)


def _masked_parameter(block: re.Match) -> str:
    opening, body, closing = block.groups()
    value = body.strip()
    lead = body[:len(body) - len(body.lstrip())]
    trail = body[len(body.rstrip()):] if value else ""
    return f"{opening}{lead}{spec_label(value)}{trail}{closing}"


def _mask_line(line: str) -> str:
    head, _, tail = line.partition(CALL)
    end = tail.rfind(")")
    spec, rest = (tail[:end], tail[end:]) if end >= 0 else (tail, "")
    inner = spec.strip()
    quote = inner[0] if inner[:1] in _QUOTES and inner else ""
    if quote:
        inner = inner[1:-1] if inner.endswith(quote) and len(inner) > 1 else inner[1:]
    return f"{head}{CALL}{quote}{spec_label(inner)}{quote}{rest}"
