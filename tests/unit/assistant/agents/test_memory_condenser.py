from unittest.mock import Mock, call

import pytest

from assistant.agents.MemoryCondenser import MemoryCondenser, split_sections, word_count

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


class TestCondense:
    def test_every_section_is_sent_with_the_running_tally_and_the_whole_bank_in_the_system_prompt(self):
        sections = split_sections(BANK)
        # Marcus loses 3 words, the others come back as they were.
        replies = [f"<section>{s}</section>" for s in sections]
        replies[2] = "<section>## Marcus\n* coffee\n### Work\n* at the lab</section>"
        llm = _llm(replies)
        new_bank, outcome = MemoryCondenser(llm, word_limit=100).condense(BANK, cut=10)
        systems = [c.kwargs['system_message'] for c in llm.think.call_args_list]
        assert len(set(systems)) == 1 and BANK in systems[0] and "At least 10 words must go" in systems[0]
        messages = [c.kwargs['user_message'] for c in llm.think.call_args_list]
        assert "Section 1 of 5" in messages[0] and "removed 0 words, you need to remove 10 more" in messages[0]
        assert "4 more sections to go" in messages[0]
        assert "removed 3 words, you need to remove 7 more" in messages[3] and sections[3] in messages[3]
        assert "0 more sections to go" in messages[4]
        # the share: the remaining cut in proportion to the section's size among those still to come
        sizes = [word_count(s) for s in sections]
        assert f"about {round(10 * sizes[0] / sum(sizes))} words" in messages[0]
        assert f"about {round(7 * sizes[3] / (sizes[3] + sizes[4]))} words" in messages[3]
        assert "about 7 words" in messages[4]       # the last section hears the whole remainder
        assert "* coffee\n" in new_bank and "hates tea" not in new_bank and new_bank.endswith("* two floors\n")
        assert outcome["to"] == word_count(BANK) - 3 and outcome['from'] == word_count(BANK)
        assert outcome['sections'] == 5

    def test_the_remaining_ask_never_goes_negative(self):
        llm = _llm(["<section># Only</section>"])
        MemoryCondenser(llm, word_limit=100).condense("# Only\nten words here to be cut right away now ok", cut=3)
        assert "remove 3 more" in llm.think.call_args.kwargs['user_message']

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
