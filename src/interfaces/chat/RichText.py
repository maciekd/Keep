"""The one rich-text dialect every chat interface speaks: Telegram's HTML subset.

The assistant produces it (the prompt rules say so), the chat contract carries it verbatim,
and each transport renders it — Telegram passes it through, the TUI parses it, a platform
without HTML converts it. This module is the dialect's single definition: which tags exist,
how to sanitize LLM output down to them, where a streaming prefix can be cut without leaving
a tag open, how to split long messages, and how to render markdown into it.
"""
import re
from html import escape, unescape

from common.logging import info, trace

ALLOWED_TAGS = {'b', 'i', 'u', 's', 'a', 'code', 'pre', 'strong', 'em', 'del', 'strike',
                'tg-spoiler', 'blockquote', 'tg-emoji'}
_TAG_RE = re.compile(r'<(/?)([\w-]+)([^>]*)>')
_PRE_TAG_RE = re.compile(r'<(/?)pre\b', re.IGNORECASE)
_TABLE_SEPARATOR_RE = re.compile(r'^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$')
_NUMERIC_CELL_RE = re.compile(r'^[\d.,%$+\-–]+$')


def command_line(text: str, detail: str = "") -> str:
    """The "• doing X" command echo; *detail* is the completion note shown after it."""
    line = f" • <i>{escape(text)}</i>"
    return f"{line} <i>({escape(detail)})</i>" if detail else line


def sanitize(text: str) -> str:
    """Strip HTML tags outside the dialect, keeping only allowed ones."""
    return _TAG_RE.sub(lambda m: m.group(0) if m.group(2).lower() in ALLOWED_TAGS else '', text)


def strip_tags(text: str) -> str:
    """Remove all HTML tags, returning plain text."""
    return _TAG_RE.sub('', text)


def safe_prefix_end(text: str) -> int:
    """End of the longest prefix in which every allowed tag is balanced — the most of a
    streaming buffer that can be shown without an open tag breaking the render."""
    incomplete = re.search(r'<[^>]*$', text)
    check_end = incomplete.start() if incomplete else len(text)

    tags = []
    for m in _TAG_RE.finditer(text[:check_end]):
        tag = m.group(2).lower()
        if tag not in ALLOWED_TAGS:
            continue
        tags.append((m.start(), tag, bool(m.group(1))))

    if not tags:
        return check_end

    matched: set[int] = set()
    close_stack: list[tuple[int, str]] = []

    for i in range(len(tags) - 1, -1, -1):
        _start, name, is_closing = tags[i]
        if is_closing:
            close_stack.append((i, name))
        else:
            for j in range(len(close_stack) - 1, -1, -1):
                if close_stack[j][1] == name:
                    matched.add(i)
                    matched.add(close_stack[j][0])
                    close_stack.pop(j)
                    break

    for i, (start, name, is_closing) in enumerate(tags):
        if not is_closing and i not in matched:
            return start

    return check_end


def split(text: str, max_length: int = 4096) -> list[str]:
    """Split at newlines (then spaces, then hard) so no chunk exceeds max_length."""
    if len(text) <= max_length:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        # A break at index 0 would yield an empty chunk and (for a space) never consume any
        # text, so it counts as no break point at all.
        split_at = text.rfind('\n', 0, max_length)
        if split_at < 1:
            split_at = text.rfind(' ', 0, max_length)
        if split_at < 1:
            split_at = max_length
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip('\n')
    return chunks


_MD_BOLD_RE = re.compile(r'\*\*([^*\n]+?)\*\*')
_MD_CODE_RE = re.compile(r'`([^`<>\n]{1,80})`')


def repair_markdown(text: str) -> str:
    """Safety net for a reply that slipped into Markdown despite the prompt: completed
    `**bold**` pairs and `code` spans become the dialect's tags. The text is otherwise
    already HTML, so nothing is escaped, and an unclosed marker (still streaming, or a
    literal asterisk) is left alone. Idempotent: applying it to a growing stream converts
    each pair once, when it closes."""
    text = _MD_BOLD_RE.sub(r'<b>\1</b>', text)
    return _MD_CODE_RE.sub(r'<code>\1</code>', text)


def markdown_to_html(md: str) -> str:
    """Render markdown (headings, lists, code, bold, inline code) into the dialect. A pipe
    table stays a pipe table, its cells' inline markdown converted: the dialect has no table
    tag, the console pane draws pipe rows as an aligned table (`widgets/ChatTable.py`), and
    Telegram shows the rows as the model itself writes them."""
    lines = md.split("\n")
    result = []
    in_code_block = False

    for line in lines:
        if line.startswith("```"):
            if in_code_block:
                result.append("</code></pre>")
                in_code_block = False
            else:
                in_code_block = True
                result.append("<pre><code>")
            continue

        if in_code_block:
            result.append(escape(line))
            continue

        stripped = line.strip()

        if stripped.startswith("|"):
            result.append(_inline_markdown(stripped))
            continue

        if stripped in ("---", "***", "___"):
            result.append("———")
            continue

        for marker in ("### ", "## ", "# "):
            if stripped.startswith(marker):
                result.append(f"<b>{_inline_markdown(stripped[len(marker):])}</b>")
                break
        else:
            if stripped.startswith("- ") or stripped.startswith("* "):
                result.append(f"• {_inline_markdown(stripped[2:])}")
                continue

            m = re.match(r'^(\d+)\.\s+(.+)', stripped)
            if m:
                result.append(f"{m.group(1)}. {_inline_markdown(m.group(2))}")
                continue

            result.append(_inline_markdown(line))

    if in_code_block:
        result.append("</code></pre>")

    return "\n".join(result)


def tables_as_pre(text: str, max_length: int = 0) -> str:
    """The pipe tables of a reply as `<pre>` blocks, for a platform that cannot draw a table:
    a run of consecutive lines starting with `|` becomes one block whose cells are padded to
    their column (numbers right-aligned), the separator row dropped. Tags inside a cell are
    stripped, since Telegram allows none inside `<pre>`; a line inside an existing `<pre>`
    block is left alone, as `widgets/ChatTable.py` leaves it. With `max_length` — the
    platform's message limit, which the text was already split to fit — a text the padding
    would push past it comes back untouched, the rows as the model wrote them: a wide column
    pads every row to its width, and a message over the limit is one the platform refuses."""
    out: list[str] = []
    rows: list[str] = []
    tables = 0

    def flush():
        nonlocal tables
        if rows:
            out.append(_aligned_pre(rows))
            tables += 1
            rows.clear()

    pre_depth = 0
    for line in text.split("\n"):
        if pre_depth == 0 and line.lstrip().startswith("|"):
            rows.append(line)
        else:
            flush()
            out.append(line)
        for tag in _PRE_TAG_RE.finditer(line):
            pre_depth = max(0, pre_depth + (-1 if tag.group(1) else 1))
    flush()
    if not tables:
        return text
    rendered = "\n".join(out)
    if max_length and len(rendered) > max_length:
        info(f"RichText: the padded table(s) would make the message {len(rendered)} chars, over the limit of "
             f"{max_length} — the {tables} table(s) stay pipe rows")
        return text
    trace(f"RichText: {tables} pipe table(s) rendered as <pre> blocks")
    return rendered


def _aligned_pre(lines: list[str]) -> str:
    rows = [_table_cells(line) for line in lines if not _TABLE_SEPARATOR_RE.match(line.strip())]
    if not rows:
        return "\n".join(lines)
    columns = max(len(row) for row in rows)
    widths = [max(len(unescape(row[i])) if i < len(row) else 0 for row in rows) for i in range(columns)]
    padded = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            fill = widths[i] - len(unescape(cell))
            cells.append(" " * fill + cell if _NUMERIC_CELL_RE.match(cell) else cell + " " * fill)
        padded.append("  ".join(cells).rstrip())
    return "<pre>" + "\n".join(padded) + "</pre>"


def _table_cells(line: str) -> list[str]:
    inner = strip_tags(line.strip()).replace("\\|", "\x00")
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") else inner
    return [c.strip().replace("\x00", "|") for c in inner.split("|")]


def _inline_markdown(text: str) -> str:
    text = escape(text)
    text = re.sub(r'`([^`]{1,80})`', r'<code>\1</code>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return text
