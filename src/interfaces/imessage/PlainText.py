"""Turns a reply written in the chat layer's HTML dialect into what a Messages bubble shows: plain text.

Messages renders no markup, so a tag or a Markdown mark left in a reply would be read as
typed. `render` keeps the words — a link keeps its address, a list its items, a code block
its lines — and `split` cuts a long reply into bubbles where a person would: between
paragraphs, else lines, else sentences.

Only what is markup beyond doubt is taken out, since a bubble with a mark left in it is ugly
and one with a character missing is wrong: code is never touched (`__init__.py` and `2**3`
stay as written), a tag is a tag only when its name is a known one and what follows the name
is attributes (`a<b and c>d` is text), and an emphasis mark only when the pair brackets
whole words. `__` and `_` are never emphasis here: in what an assistant writes they are
far more often part of a name.
"""
import html
import re
import unicodedata

from interfaces.chat import RichText

BUBBLE_CHARS = 1200     # the longest bubble: a screen and a half on a phone

# What may follow a tag's name: nothing, or attributes with a value. A word without one
# ("<b and c>") is no attribute, but for the one the dialect has (`<blockquote expandable>`)
# and the few HTML itself writes bare (`<td nowrap>`), none of them a word of a sentence.
_ATTRIBUTES = r"""(?:\s+[\w:-]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'<>=]+)|\s+(?:expandable|nowrap|reversed|compact))*\s*/?"""
_INLINE_TAGS = sorted(RichText.ALLOWED_TAGS - {"a", "pre", "code", "blockquote"}
                      | {"ins", "span", "mark", "small", "sub", "sup", "font", "thead", "tbody", "tfoot", "td", "th"})
_BLOCK_TAGS = ("p", "div", "blockquote", "ul", "ol", "table", "tr", "h[1-6]", "pre")


def _tag(names, closing: str = "/?") -> re.Pattern:
    return re.compile(rf"<{closing}(?:{'|'.join(names)}){_ATTRIBUTES}>", re.IGNORECASE)


_CODE_RE = re.compile(rf"<pre{_ATTRIBUTES}>(?P<pre>.*?)</pre\s*>"
                      rf"|<code{_ATTRIBUTES}>(?P<code>.*?)</code\s*>"
                      r"|^```[\w+-]*[ \t]*\n(?P<fence>.*?)^```[ \t]*$"
                      r"|`(?P<tick>[^`\n]+)`", re.IGNORECASE | re.DOTALL | re.MULTILINE)
_CODE_TAG_RE = _tag(("pre", "code"))
_LINK_OPEN = r"""<a\s[^<>]*?href\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'<>]+))[^<>]*>"""
_LINK_RE = re.compile(rf"{_LINK_OPEN}((?:(?!<a\s).)*?)</a\s*>", re.IGNORECASE | re.DOTALL)
# A link that is never closed: its text is what stands up to the end of the line or of the block.
_OPEN_LINK_RE = re.compile(rf"{_LINK_OPEN}((?:(?!</?(?:p|div|blockquote|ul|ol|li|table|tr|td|th|br|h[1-6]|pre|a)\b)[^\n])*)",
                           re.IGNORECASE)
# A character reference is one only with its semicolon: `&region=SE` in an address is not "®ion=SE".
_REFERENCE_RE = re.compile(r"&(?:#\d+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")
_MD_LINK_RE = re.compile(r"!?\[([^\[\]\n]+)\]\(\s*((?:https?://|mailto:)[^\s()]+)\s*\)")
_IMAGE_RE = _tag(("img",), closing="")
_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_ITEM_RE = _tag(("li",), closing="")
_ITEM_END_RE = re.compile(r"</li\s*>", re.IGNORECASE)
_CELLS_RE = re.compile(rf"</t[dh]\s*>\s*<t[dh]{_ATTRIBUTES}>", re.IGNORECASE)
_PARAGRAPH_END_RE = re.compile(r"</(?:p|h[1-6])\s*>", re.IGNORECASE)
_BLOCK_RE = _tag(_BLOCK_TAGS)
_INLINE_RE = _tag([re.escape(name) for name in _INLINE_TAGS] + ["a", "code", "li"])
_HEADING_RE = re.compile(r"^#{1,6}[ \t]+(?=\S)", re.MULTILINE)
_BULLET_RE = re.compile(r"^([ \t]*)\*[ \t]+(?=\S)", re.MULTILINE)
# A pair of marks is emphasis when it stands where a word begins and ends: nothing of a word
# or of another mark outside it, no space just inside it.
_BOLD_RE = re.compile(r"(?<![\w*])(\*{2,3})(?=[^\s*])([^*\n]+?)(?<=[^\s*])\1(?![\w*])")
_ITALIC_RE = re.compile(r"(?<![\w*])\*(?=\w)([^*\n]+?)(?<=[\w.!?,;:)])\*(?![\w*])")
_STRIKE_RE = re.compile(r"(?<![\w~])~~(?=[^\s~])([^~\n]+?)(?<=[^\s~])~~(?![\w~])")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SENTENCE_ENDS = (". ", "! ", "? ", "… ")
_BREAK = "\x01"         # where a block began or ended: a line break, once the tags are gone
_BREAKS_RE = re.compile(rf"[ \t\n]*{_BREAK}[ \t\n{_BREAK}]*")
_HELD_RE = re.compile("\x00(\\d+)\x00")
_JOINERS = "‍️︎⃣"      # what continues an emoji: the joiner, the variation selectors, the keycap
_SKIN_TONES = range(0x1F3FB, 0x1F400)
_REGIONAL = range(0x1F1E6, 0x1F200)


def _link(match: re.Match) -> str:
    address = next((group for group in match.group(1, 2, 3) if group is not None), "").strip()
    text = _INLINE_RE.sub("", match.group(4)).strip()
    if not text or text == address:
        return address
    return f"{text} ({address})" if address else text


def _unescape(text: str) -> str:
    return _REFERENCE_RE.sub(lambda match: html.unescape(match.group(0)), text)


def render(text: str) -> str:
    """The plain text of a reply in the HTML dialect, with the Markdown a model slips into taken out as well."""
    held: list[str] = []

    def hold(match: re.Match) -> str:
        """Code leaves the text until every mark around it is dealt with."""
        kind = match.lastgroup
        code = match.group(kind)
        if kind in ("pre", "code"):
            code = _CODE_TAG_RE.sub("", code)
        held.append(_unescape(code).strip("\n") if kind in ("pre", "fence") else _unescape(code))
        place = f"\x00{len(held) - 1}\x00"
        return f"{_BREAK}{place}{_BREAK}" if kind in ("pre", "fence") else place

    text = _CODE_RE.sub(hold, (text or "").replace("\x00", "").replace(_BREAK, ""))
    text = _LINK_RE.sub(_link, text)
    text = _OPEN_LINK_RE.sub(_link, text)
    text = _IMAGE_RE.sub("", text)
    text = _BREAK_RE.sub("\n", text)
    text = _ITEM_RE.sub(f"{_BREAK}- ", text)
    text = _ITEM_END_RE.sub(_BREAK, text)
    text = _CELLS_RE.sub(" | ", text)
    text = _PARAGRAPH_END_RE.sub(f"\n\n{_BREAK}", text)
    text = _BLOCK_RE.sub(_BREAK, text)
    text = _unescape(_INLINE_RE.sub("", text))
    text = _BREAKS_RE.sub(lambda run: "\n\n" if run.group(0).count("\n") >= 2 else "\n", text)
    text = _MD_LINK_RE.sub(lambda m: m.group(2) if m.group(1).strip() == m.group(2) else f"{m.group(1).strip()} ({m.group(2)})",
                           text)
    text = _HEADING_RE.sub("", text)
    text = _BULLET_RE.sub(r"\1- ", text)
    text = _BOLD_RE.sub(r"\2", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _STRIKE_RE.sub(r"\1", text)
    text = _BLANK_LINES_RE.sub("\n\n", "\n".join(line.rstrip() for line in text.split("\n"))).strip()
    text = _HELD_RE.sub(lambda m: held[int(m.group(1))], text)      # last: code keeps its indentation and its blank lines
    visible = any(not char.isspace() and unicodedata.category(char) != "Cf" for char in text)
    return text if visible else ""       # a zero-width space alone would be sent as an empty bubble


def _continues(char: str) -> bool:
    """Whether *char* belongs to the character before it: a joiner, a variation selector, a
    skin tone, a combining mark."""
    return char in _JOINERS or ord(char) in _SKIN_TONES or unicodedata.combining(char) > 0


def _whole(text: str, cut: int, floor: int) -> int:
    """*cut* moved back, as far as *floor*, so that it does not fall inside one character as
    people see it: an emoji joined of several (a family, a flag, a skin tone), a letter with
    its accents."""
    at = cut
    while at > floor:
        before, after = text[at - 1], text[at]
        if _continues(after) or before == "‍":
            at -= 1
            continue
        if ord(before) in _REGIONAL and ord(after) in _REGIONAL:
            start = at - 1
            while start > 0 and ord(text[start - 1]) in _REGIONAL:
                start -= 1
            if (at - start) % 2:        # an odd number of flag letters before the cut: it halves a flag
                at -= 1
                continue
        return at
    return cut


def split(text: str, limit: int = BUBBLE_CHARS) -> list[str]:
    """*text* as bubbles of at most *limit* characters, cut at the last paragraph break that
    fits, else the last line break, sentence end or space, else mid-word."""
    bubbles = []
    text = text.rstrip().lstrip("\n")      # the first line's indentation may be a code block's
    floor = max(1, limit // 4)      # no bubble of a few words in front of a full one
    while len(text) > limit:
        cut = text.rfind("\n\n", floor, limit)
        if cut < floor:
            cut = text.rfind("\n", floor, limit)
        if cut < floor:
            cut = max(text.rfind(end, floor, limit) + 1 for end in _SENTENCE_ENDS)     # the mark stays with its sentence
        if cut < floor:
            cut = text.rfind(" ", floor, limit)
        if cut < floor:
            cut = _whole(text, limit, floor)
        if text[:cut].strip():
            bubbles.append(text[:cut].rstrip())
        text = text[cut:]
        text = text.lstrip("\n") if text.startswith("\n") else text.lstrip()
    if text:
        bubbles.append(text)
    return bubbles
