"""Makes what somebody outside the household chose — a group's name, a contact's name, the words of a message — safe to stand next to the model's instructions.

Anyone can name a group chat, write into one, or be quoted by an owner's swipe-reply, and all
of it ends up in text the model reads: the destination list of every prompt, the note in front
of a turn, the backlog of a chat. Three forms, by where the text goes:

- `safe_name` — a name that is shown as a name: letters, digits and single spaces, a few marks
  that close nothing, 40 characters. No quote, bracket, colon or line break survives, nor a
  letter that only looks like one, so a name cannot end the quotes or the sentence it is put in.
- `quoted` — a message kept as context: one line, as a JSON string. Whatever it says, a parser
  and a model see one string; the frame around the backlog cannot be closed from inside.
- `inline` — a quote that goes into a frame of the chat layer's own (`[Replying to: "…"]`):
  one line, with the frame's delimiters taken out.
"""
import json
import re
import unicodedata

from assistant.PromptInjectionDetector import is_prompt_injection
from common.logging import info

NAME_CHARS = 40
# Marks a name may keep: none of them opens or closes anything, starts a label ("Name:") or
# ends a line — a hyphen, an underscore, "&" and "+" join words ("Tom & Anna", "Team-A"), and
# the sentence marks are in group names as people write them ("Go, team!").
NAME_MARKS = "-_&+.,!?"
HANDLE_CHARS = 64
_NOT_IN_A_HANDLE = re.compile(r"[^\w.+@-]")
_APOSTROPHES = "'’ʼʻ"
MARKS_PER_LETTER = 3            # accents one letter may carry: more is a tower of marks, not a letter
# Letters by Unicode's count that read as something else. The modifier letters (category Lm) go
# as a class — "ʺ" and "ˮ" are quotes to a reader, "ː" and "ꓽ" a colon — but for the few that
# words are spelled with (々, ー, ๆ). Among the ordinary letters: the Canadian syllabics shaped
# like angle brackets, a slash and a tick, the click letters shaped like a bar and an
# exclamation mark, and the fillers, which show as nothing.
_MODIFIERS_OF_WORDS = frozenset("々ーｰๆໆៗ")
_NOT_A_LETTER = frozenset("ᐸᐳᑉᐟᐠᑊǀǁǂǃ" "\u115f\u1160\u3164\uffa0")
# Marks with nothing to show: the variation selectors, the grapheme joiner, Khmer's inherent vowels.
_UNSEEN_MARKS = frozenset("\u034f\u17b4\u17b5\u180b\u180c\u180d\u180f") | frozenset(
    map(chr, (*range(0xFE00, 0xFE10), *range(0xE0100, 0xE01F0))))
# Direction overrides and isolates: they reorder what is shown without changing what is there.
_DIRECTION_MARKS = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮⁦⁧⁨⁩"))
_BRACKETS = str.maketrans({"[": "(", "]": ")", "［": "(", "］": ")", "【": "(", "】": ")", "〔": "(", "〕": ")",
                           "〚": "(", "〛": ")", "⟦": "(", "⟧": ")"})
_INLINE = str.maketrans({'"': "'", "“": "'", "”": "'", "„": "'", "＂": "'", "\\": "/"})


def safe_name(raw) -> str:
    """*raw* as a name that can be shown to the model: its letters, digits and `NAME_MARKS`,
    words one space apart, `NAME_CHARS` at most. "" when nothing usable is left — no letter or
    digit — or when what is left reads as an instruction; the caller then names the thing
    by what it knows itself."""
    kept, marks = "", 0
    for char in unicodedata.normalize("NFC", str(raw or "")):
        if char in _APOSTROPHES:
            continue        # "Mom's" is "Moms", not two words
        category = unicodedata.category(char)
        if category[0] == "M":
            # a combining mark belongs to the letter in front of it, and goes with whatever went;
            # one that encloses its letter (a keycap, a circle) or shows nothing is no accent
            marks += 1
            if (kept[-1:].strip() and kept[-1] not in NAME_MARKS and marks <= MARKS_PER_LETTER
                    and category != "Me" and char not in _UNSEEN_MARKS):
                kept += char
            continue
        marks = 0
        letter = (category[0] in "LN" and char not in _NOT_A_LETTER
                  and (category != "Lm" or char in _MODIFIERS_OF_WORDS))
        kept += char if letter or char in NAME_MARKS else " "
    name = " ".join(kept.split())[:NAME_CHARS].strip(" -_&+.,")
    if not any(char.isalnum() for char in name):
        return ""
    if reads_as_injection(name):
        info(f"iMessage chat: a name that reads as an instruction is not used ({len(name)} characters)")
        return ""
    return name


def reads_as_injection(text) -> bool:
    """Whether *text* looks like a prompt injection as it is written or as a reader takes it:
    the look-alike letters as the plain ones (full-width, mathematical), accents aside, and
    whatever joins the words — an underscore, a dot — as the space it stands for."""
    text = str(text or "")
    plain = "".join(char if char.isalnum() else " " for char in unicodedata.normalize("NFKD", text)
                    if not unicodedata.combining(char))
    return is_prompt_injection(text) or is_prompt_injection(" ".join(plain.split()))


def safe_handle(raw) -> str:
    """A phone number or an email as chat.db has it, with nothing in it but what a handle is made of."""
    return _NOT_IN_A_HANDLE.sub("", str(raw or ""))[:HANDLE_CHARS]


def single_line(text, limit: int) -> str:
    """*text* as one line of at most *limit* characters: every line break and separator a
    space (U+2028, U+2029, vertical tab and form feed included), control characters and
    direction marks gone, square brackets of any width round ones."""
    line = " ".join(str(text or "").translate(_DIRECTION_MARKS).split())
    line = "".join(char for char in line if unicodedata.category(char) != "Cc").translate(_BRACKETS)
    return line if len(line) <= limit else line[:limit].rstrip() + "…"


def quoted(text, limit: int) -> str:
    """*text* as one line holding one JSON string."""
    return json.dumps(single_line(text, limit), ensure_ascii=False)


def inline(text, limit: int) -> str:
    """*text* for a frame that puts it between double quotes itself: one line, no double quote
    and no backslash left to end the frame's own."""
    return single_line(text, limit).translate(_INLINE)
