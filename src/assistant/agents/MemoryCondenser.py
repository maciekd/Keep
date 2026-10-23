"""Shortens an oversize memory bank one section at a time, with the whole bank in view and a running tally.

A single "condense this bank" call does not work: the model trims every line by a few percent,
cannot measure what it has cut, and stops far short of the ask (three tries on a 5000-word
bank removed 236–347 of the 992 words asked). Here the model sees the whole bank once, in a
system prompt that is cached, and then gets the bank back section by section, each time with
what it has removed so far and what is still to go — the measurement it lacks. With the cache
the pass costs little more than the one call did.
"""
import re
import time
from datetime import datetime

from common.logging import info, trace
from ..llm.AbstractLLM import Thinking

PROMPT = """
You are condensing the long-term memory bank of a home automation AI system serving a family. The bank is in front
of the AI in every conversation, and has a size limit of {WORD_LIMIT} words. It is OVER the limit: it is {WORDS}
words. Today is {TODAY}.

Your one job is to make it shorter! At least {CUT} words must go, so that the bank ends up no longer than {TARGET}
words. What to cut and how is your call.
 * condense the same facts into fewer words
 * merge repeated points
 * shorten a story to its core
 * REMOVE text and facts that are the least important

## The memory bank:
{CURRENT_MEMORY}

You have now seen the entire memory bank. We go over it one section at a time: you get a section, you return it
condensed. Keep the heading line as it is. A section can shrink to its heading alone when nothing in it is worth
its place.

## Output format
The condensed section (markdown, heading included) between <section> and </section>. Nothing else.
"""

SECTION_MESSAGE = """Section {N} of {TOTAL} ({WORDS} words). You have currently removed {REMOVED} words, you need to \
remove {REMAINING} more! {LEFT} more sections to go after this one. This section's fair share of what is left is \
about {SHARE} words.

<section>
{SECTION}
</section>"""

# A section starts at a heading of level 1 or 2 — level 3 and below stay inside their section.
HEADING = re.compile(r'^#{1,2} ')


def word_count(text: str) -> int:
    return len(text.split())


def split_sections(bank: str) -> list[str]:
    """The bank cut at its level 1 and 2 headings, each piece a heading and what follows it,
    the text before the first heading (if any) a piece of its own. `"\\n".join` gives the
    bank back."""
    sections, current = [], []
    for line in bank.splitlines():
        if HEADING.match(line) and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current))
    return sections


class MemoryCondenser:
    def __init__(self, llm, word_limit: int):
        self.llm = llm
        self.word_limit = word_limit

    def condense(self, bank: str, cut: int) -> tuple[str, dict]:
        """The bank with at least `cut` words asked off it, section by section, and what
        happened: the words before and after and the time. A section whose reply was cut off
        at the output cap, holds no section, or is no shorter than the original is kept as it
        was."""
        words = word_count(bank)
        system = PROMPT.replace("{WORD_LIMIT}", str(self.word_limit)).replace("{WORDS}", str(words)) \
            .replace("{CUT}", str(cut)).replace("{TARGET}", str(words - cut)) \
            .replace("{TODAY}", f"{datetime.now():%Y-%m-%d}").replace("{CURRENT_MEMORY}", bank)
        sections = split_sections(bank)
        info(f"Condenser: {words} words in {len(sections)} sections, {cut} words to remove")
        start_time = time.time()
        removed, result, unprocessed = 0, [], words
        self.llm.set_prompt_cache(True)
        try:
            for index, section in enumerate(sections, 1):
                size = word_count(section)
                remaining = max(0, cut - removed)
                # The section's share of what is left, in proportion to its size among the
                # sections still to come: a shortfall moves on to the next ones, but each hears
                # a number sized to itself rather than the whole remainder.
                share = round(remaining * size / max(1, unprocessed))
                message = SECTION_MESSAGE.replace("{N}", str(index)).replace("{TOTAL}", str(len(sections))) \
                    .replace("{WORDS}", str(size)).replace("{REMOVED}", str(removed)) \
                    .replace("{REMAINING}", str(remaining)).replace("{SHARE}", str(share)) \
                    .replace("{LEFT}", str(len(sections) - index)).replace("{SECTION}", section)
                unprocessed -= size
                response = self.llm.think(system_message=system, user_message=message, thinking=Thinking.FULL)
                condensed = self._section(response)
                heading = section.splitlines()[0][:60]
                if getattr(self.llm, 'last_stop_reason', None) == 'max_tokens':
                    info(f"Condenser: section {index}/{len(sections)} {heading!r} — the reply was cut off, kept as it was")
                    condensed = section
                elif condensed is None:
                    info(f"Condenser: section {index}/{len(sections)} {heading!r} — no section in the reply, kept as it was")
                    condensed = section
                elif word_count(condensed) >= size:
                    trace(f"Condenser: section {index}/{len(sections)} {heading!r} came back no shorter — kept as it was")
                    condensed = section
                else:
                    info(f"Condenser: section {index}/{len(sections)} {heading!r} {size} → {word_count(condensed)} words")
                removed += size - word_count(condensed)
                result.append(condensed)
        finally:
            self.llm.set_prompt_cache(False)
        new_bank = "\n".join(result).rstrip("\n") + "\n"
        outcome = {'from': words, 'cut': cut, 'to': word_count(new_bank), 'sections': len(sections),
                   'llm_duration_seconds': time.time() - start_time}
        info(f"Condenser: {words} → {outcome['to']} words ({removed} removed of {cut} asked) in "
             f"{outcome['llm_duration_seconds']:.0f}s")
        return new_bank, outcome

    @staticmethod
    def _section(response: str) -> str | None:
        """The section in the reply, None when there is none. An unclosed block runs to the end."""
        match = re.search(r'<section>(.*?)(?:</section>|\Z)', response, re.DOTALL)
        if not match or not match.group(1).strip():
            return None
        return match.group(1).strip("\n")
