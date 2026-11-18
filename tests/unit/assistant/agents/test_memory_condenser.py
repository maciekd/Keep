import re
from unittest.mock import Mock, call

import pytest

from assistant.llm.AbstractLLM import LLMRefusalError
from assistant.llm.LLMUsage import LLMUsage
from assistant.agents.MemoryCondenser import SHARE_CEILING, MemoryCondenser, split_sections, word_count

BANK = """Intro line before any heading.
# People
## Marcus
* likes coffee, hates tea
### Work
* at the lab
## Charlotte
* paints
# House
* two floors
"""


def _llm(replies):
    llm = Mock()
    llm.last_stop_reason = None
    llm.think.side_effect = replies
    return llm


class TestSplitSections:
    def test_cuts_at_level_one_and_two_headings_and_joins_back_to_the_bank(self):
        sections = split_sections(BANK)
        assert [s.splitlines()[0] for s in sections] == ["Intro line before any heading.", "# People", "## Marcus",
                                                         "## Charlotte", "# House"]
        assert "### Work" in sections[2]          # a level-3 heading stays inside its section
        assert "\n".join(sections) + "\n" == BANK

    def test_a_bank_without_headings_is_one_section(self):
        assert split_sections("just text\nmore text") == ["just text\nmore text"]


class TestALongSectionKeptAsItWas:
    def test_its_cut_is_not_handed_to_the_short_sections_after_it(self):
        """prod 2026-11-12: 2,599 words came back cut off and a 126-word section after it was
        told to lose 112. No section is asked for much more than its own part of the cut."""
        long, short = "## Marcus\n" + "word " * 2600, "# Facts\n" + "fact " * 124
        bank = f"{long}\n{short}\n"
        llm = _llm([f"<section>{long}</section>", f"<section>{short}</section>"])
        cut = 700
        MemoryCondenser(llm, word_limit=2000).condense(bank, cut=cut)
        message = llm.think.call_args_list[1].kwargs['user_message']
        size, words = word_count(short), word_count(bank)
        asked = [int(n) for n in re.findall(r"(\d+) (?:are still to go|words\.\s*$)", message, re.MULTILINE)]
        assert len(asked) == 2, message
        assert all(n <= 2 * cut * size / words for n in asked), message


class TestCondense:
    def test_every_section_is_sent_with_the_running_tally_and_the_whole_bank_in_the_system_prompt(self):
        sections = split_sections(BANK)
        # Marcus loses 3 words, the others come back as they were.
        replies = [f"<section>{s}</section>" for s in sections]
        replies[2] = "<section>## Marcus\n* coffee\n### Work\n* at the lab</section>"
        llm = _llm(replies)
        new_bank, outcome = MemoryCondenser(llm, word_limit=100).condense(BANK, cut=10)
        systems = [c.kwargs['system_message'] for c in llm.think.call_args_list]
        assert len(set(systems)) == 1 and BANK in systems[0] and "About 10 words should go" in systems[0]
        messages = [c.kwargs['user_message'] for c in llm.think.call_args_list]
        assert "Section 1 of 5" in messages[0] and "0 words are removed and about 10 are still to go" in messages[0]
        assert "4 more sections to go" in messages[0]
        assert "3 words are removed and about " in messages[3] and sections[3] in messages[3]
        assert "0 more sections to go" in messages[4]
        # the share: the remaining cut in proportion to the section's size among those still to come
        sizes = [word_count(s) for s in sections]
        assert f"roughly {round(10 * sizes[0] / sum(sizes))} words" in messages[0]
        # ... and after a shortfall never the whole remainder: at most the ceiling over its own part of the cut
        for index in (3, 4):
            ceiling = round(sizes[index] * 10 * SHARE_CEILING / sum(sizes))
            assert f"roughly {ceiling} words" in messages[index] and ceiling < 7
        assert "* coffee\n" in new_bank and "hates tea" not in new_bank and new_bank.endswith("* two floors\n")
        assert outcome["to"] == word_count(BANK) - 3 and outcome['from'] == word_count(BANK)
        assert outcome['sections'] == 5

    def test_the_remaining_ask_never_goes_negative(self):
        llm = _llm(["<section># Only</section>"])
        MemoryCondenser(llm, word_limit=100).condense("# Only\nten words here to be cut right away now ok", cut=3)
        assert "about 3 are still to go" in llm.think.call_args.kwargs['user_message']

    @pytest.mark.parametrize("reply", ["no tags at all", "<section>   </section>", "<section>\n</section>"])
    def test_a_reply_without_a_section_keeps_the_section(self, reply):
        llm = _llm([reply, "<section># House\n* two floors</section>"])
        new_bank, _ = MemoryCondenser(llm, word_limit=100).condense("# People\n* Marcus likes coffee\n# House\n* two floors\n", cut=2)
        assert new_bank == "# People\n* Marcus likes coffee\n# House\n* two floors\n"

    def test_a_reply_no_shorter_than_the_section_is_not_taken(self):
        llm = _llm(["<section># People\n* Marcus likes coffee a lot more</section>"])
        new_bank, outcome = MemoryCondenser(llm, word_limit=100).condense("# People\n* Marcus likes coffee\n", cut=2)
        assert new_bank == "# People\n* Marcus likes coffee\n" and outcome['to'] == outcome['from']

    def test_a_reply_cut_off_at_the_output_cap_keeps_the_section(self):
        llm = _llm(["<section># People\n* Marc"])
        llm.last_stop_reason = 'max_tokens'
        new_bank, _ = MemoryCondenser(llm, word_limit=100).condense("# People\n* Marcus likes coffee\n", cut=2)
        assert new_bank == "# People\n* Marcus likes coffee\n"

    def test_an_unclosed_section_from_a_finished_reply_is_taken(self):
        llm = _llm(["<section># People\n* coffee"])
        new_bank, _ = MemoryCondenser(llm, word_limit=100).condense("# People\n* Marcus likes coffee\n", cut=2)
        assert new_bank == "# People\n* coffee\n"

    def test_the_cache_is_on_for_the_pass_and_off_after_even_when_a_call_fails(self):
        llm = _llm([RuntimeError("down")])
        with pytest.raises(RuntimeError):
            MemoryCondenser(llm, word_limit=100).condense("# People\n* Marcus\n", cut=1)
        assert llm.set_prompt_cache.call_args_list == [call(True), call(False)]


def test_an_empty_bank_makes_no_call_and_counts_zero_words():
    llm = _llm([])
    _, outcome = MemoryCondenser(llm, 100).condense("", cut=5)
    llm.think.assert_not_called()
    assert outcome['sections'] == 0 and outcome['to'] == 0


class TestWhatThePassUsed:
    def test_the_outcome_carries_the_sum_of_every_section_call(self):
        sections = split_sections(BANK)
        llm = _llm(["<section>short</section>"] * len(sections))
        llm.last_usage = LLMUsage(input_tokens=10, output_tokens=3, cost_usd=0.25)
        _, outcome = MemoryCondenser(llm, word_limit=10).condense(BANK, cut=5)
        assert f"{3 * len(sections)} output tokens" in outcome['usage']
        assert f"${0.25 * len(sections):.2f}" in outcome['usage']

    def test_a_backend_that_reports_no_usage_still_condenses(self):
        llm = _llm(["<section>short</section>"] * len(split_sections(BANK)))
        _, outcome = MemoryCondenser(llm, word_limit=10).condense(BANK, cut=5)
        assert outcome['to'] < outcome['from']


TWO = "# People\n* Marcus likes coffee\n# House\n* two floors up\n"
PEOPLE, HOUSE = "<section># People\n* coffee</section>", "<section># House\n* floors</section>"


def _batching(batched, ordinary=()):
    llm = _llm(list(ordinary))
    llm.think_batched.side_effect = batched
    return llm


class TestBatchedSections:
    """A condensing nobody waits for sends each section as a batch of its own, one after
    another, and stops asking the batch API at the first batch that gives no answer."""

    def test_every_section_comes_from_a_batch_with_the_tally_of_the_ones_before(self):
        llm = _batching([PEOPLE, HOUSE])
        new_bank, _ = MemoryCondenser(llm, 100, batch=True).condense(TWO, cut=4)
        assert new_bank == "# People\n* coffee\n# House\n* floors\n"
        assert "2 words are removed" in llm.think_batched.call_args_list[1].kwargs['user_message']
        llm.think.assert_not_called()

    def test_the_first_batch_without_an_answer_is_the_last_one_asked(self):
        bank = TWO + "# Car\n* a red one\n"
        llm = _batching([None], ordinary=[PEOPLE, HOUSE, "<section># Car\n* red</section>"])
        condenser = MemoryCondenser(llm, 100, batch=True)
        new_bank, _ = condenser.condense(bank, cut=6)
        assert new_bank == "# People\n* coffee\n# House\n* floors\n# Car\n* red\n"
        assert llm.think_batched.call_count == 1 and llm.think.call_count == 3
        # ... and a later pass of the same condenser asks none either
        llm.think.side_effect = ["<section># People</section>", "nothing", "nothing"]
        condenser.condense(new_bank, cut=2)
        assert llm.think_batched.call_count == 1

    def test_a_batch_that_fails_is_answered_by_the_ordinary_call(self):
        llm = _batching([RuntimeError("batches are down")], ordinary=[PEOPLE, HOUSE])
        new_bank, _ = MemoryCondenser(llm, 100, batch=True).condense(TWO, cut=4)
        assert new_bank == "# People\n* coffee\n# House\n* floors\n"

    def test_a_refused_batch_is_not_sent_again(self):
        llm = _batching([LLMRefusalError("refused")])
        with pytest.raises(LLMRefusalError):
            MemoryCondenser(llm, 100, batch=True).condense(TWO, cut=4)
        llm.think.assert_not_called()

    def test_a_batch_cut_off_before_any_text_keeps_the_section_and_is_not_asked_again(self):
        """Asked again the reply would be cut off again — and a cut-off reply is no sign of a
        slow queue, so the sections after it still go through the batch API."""
        llm = _batching(None)
        replies = iter([(None, 'max_tokens'), (HOUSE, 'end_turn')])

        def batched(**_):
            reply, llm.last_stop_reason = next(replies)
            return reply
        llm.think_batched.side_effect = batched
        new_bank, _ = MemoryCondenser(llm, 100, batch=True).condense(TWO, cut=4)
        assert new_bank == "# People\n* Marcus likes coffee\n# House\n* floors\n"
        llm.think.assert_not_called()

    def test_a_condensing_somebody_waits_for_asks_no_batch(self):
        llm = _batching([], ordinary=[PEOPLE, HOUSE])
        MemoryCondenser(llm, 100).condense(TWO, cut=4)
        llm.think_batched.assert_not_called()

    def test_the_pass_is_reported_at_batch_price_only_when_its_sections_were_batches(self):
        usage = LLMUsage(output_tokens=3, batch=True, cost_usd=0.1)
        llm = _batching([PEOPLE, HOUSE])
        llm.last_usage = usage
        _, batched = MemoryCondenser(llm, 100, batch=True).condense(TWO, cut=4)
        llm = _batching([], ordinary=[PEOPLE, HOUSE])
        llm.last_usage = LLMUsage(output_tokens=3, cost_usd=0.2)
        _, ordinary = MemoryCondenser(llm, 100).condense(TWO, cut=4)
        assert "batch" in batched['usage'] and "batch" not in ordinary['usage']
