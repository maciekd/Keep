import threading
from datetime import date, datetime, timedelta
from unittest.mock import Mock, call, patch

import pytest

from assistant import MemoryFile
from assistant.Procedures import Procedures
from assistant.agents.LongTermMemoryCurator import (CONDENSE_PASSES, CONDENSE_TRIGGER, HISTORY_SOURCE,
                                                    MEMORY_WORD_LIMIT, LongTermMemoryCurator)
from assistant.commands.Commands import is_reference_result
from common.FloxyDatabase import NO_INTERACTIONS
from tests.unit.waiting import FAILURE_BOUND


def _no_events():
    """A CalendarAccess whose week is empty."""
    access = Mock()
    access.fetch.return_value = "(No upcoming events)"
    return access


class TestLongTermMemoryCurator:
    def test_default_llm_created_with_prompt_cache_disabled(self):
        # The nightly curation call is one-shot: a prompt cache written for it is
        # never read again, so the curator must opt out of caching (the cache-write
        # surcharge on ~128k input tokens would be pure cost).
        with patch('assistant.agents.LongTermMemoryCurator.parse_llm_config',
                   return_value=('ClaudeLLM', 'some-model')), \
             patch('assistant.agents.LongTermMemoryCurator.create_llm') as mock_create:
            curator = LongTermMemoryCurator(test_mode=True)
        mock_create.assert_called_once_with('ClaudeLLM', 'some-model', prompt_cache=False)
        mock_create.return_value.initialize.assert_called_once()
        assert curator.llm is mock_create.return_value

    def test_curate_memory_with_data_passes_data_to_llm(self):
        mock_llm = Mock()
        mock_llm.think.return_value = "# Memory\nUpdated"
        curator = LongTermMemoryCurator(llm=mock_llm, test_mode=True)

        curator.curate_memory_with_data(
            interaction_history="test history",
            calendar_data="",
            current_memory="test memory"
        )

        call_args = mock_llm.think.call_args
        prompt = call_args[1].get('system_message', call_args[0][0] if call_args[0] else '')
        assert "test history" in prompt
        assert "test memory" in prompt

    def test_extract_memory_from_markdown_code_block(self):
        response = "Here is the memory:\n```markdown\n# Memory\nContent here\n```\nDone."
        result = LongTermMemoryCurator._extract_memory_content(response)
        assert result == "# Memory\nContent here"

    def test_extract_memory_from_plain_code_block(self):
        response = "```\n# Memory\nPlain block\n```"
        result = LongTermMemoryCurator._extract_memory_content(response)
        assert result == "# Memory\nPlain block"

    def test_extract_memory_from_plain_text(self):
        response = "# Memory\nJust plain text"
        result = LongTermMemoryCurator._extract_memory_content(response)
        assert result == "# Memory\nJust plain text"

    def test_validation_rejects_short_memory(self):
        mock_llm = Mock()
        # Return suspiciously short memory (less than 30% of current)
        mock_llm.think.return_value = "x"
        curator = LongTermMemoryCurator(llm=mock_llm, test_mode=True)

        current = "A" * 1000  # 1000 chars
        memory, _ = curator.curate_memory_with_data("history", "calendar", current)
        # Should keep old memory
        assert memory == current

    def test_reply_cut_off_at_max_tokens_keeps_the_previous_bank(self):
        """A truncated bank is a bank missing its tail; saving it forgets every section
        past the cut for good, so the old one stays."""
        mock_llm = Mock()
        mock_llm.think.return_value = "# Memory\n" + "A" * 900
        mock_llm.last_stop_reason = 'max_tokens'
        curator = LongTermMemoryCurator(llm=mock_llm, test_mode=True)

        current = "B" * 1000
        with patch('assistant.agents.LongTermMemoryCurator.error') as mock_error:
            memory, metadata = curator.curate_memory_with_data("history", "calendar", current)

        assert memory == current
        assert metadata['truncated'] is True
        assert "cut short" in mock_error.call_args_list[0].args[0]
        assert "unchanged" in LongTermMemoryCurator.completion_note(metadata)

    def test_finished_reply_is_not_flagged_as_truncated(self):
        mock_llm = Mock()
        mock_llm.think.return_value = "# Memory\n" + "A" * 900
        mock_llm.last_stop_reason = 'end_turn'
        curator = LongTermMemoryCurator(llm=mock_llm, test_mode=True)

        memory, metadata = curator.curate_memory_with_data("history", "calendar", "B" * 1000)
        assert memory.startswith("# Memory")
        assert metadata['truncated'] is False
        assert "memory updated" in LongTermMemoryCurator.completion_note(metadata)

    def test_validation_accepts_normal_length_memory(self):
        mock_llm = Mock()
        new_content = "# Memory\n" + "A" * 500
        mock_llm.think.return_value = new_content
        curator = LongTermMemoryCurator(llm=mock_llm, test_mode=True)

        current = "B" * 600
        memory, _ = curator.curate_memory_with_data("history", "calendar", current)
        assert memory == new_content

    def test_save_memory_writes_file(self, tmp_path):
        curator = LongTermMemoryCurator(llm=Mock(), test_mode=True,
                                         memory_file_path=str(tmp_path / "memory.md"))
        path = curator.save_memory("# Test Memory\nContent")
        assert (tmp_path / "memory.md").read_text() == "# Test Memory\nContent"
        assert path == str(tmp_path / "memory.md")

    def test_load_memory_returns_empty_for_missing(self, tmp_path):
        curator = LongTermMemoryCurator(llm=Mock(), test_mode=True,
                                         memory_file_path=str(tmp_path / "nonexistent.md"))
        assert curator._load_memory() == ""

class TestCurationPeriod:
    """curate_memory takes an explicit period so a first-run curation can cover
    "since 14:03" instead of the nightly two-day lookback."""

    def _curator(self, history="(No interactions in this period)"):
        llm = Mock()
        llm.think.return_value = "# Memory\nA bank long enough to pass the size check for sure"
        curator = LongTermMemoryCurator(llm=llm, test_mode=True)
        curator.floxy_db = Mock()
        curator.floxy_db.get_interaction_history.return_value = history
        curator._load_memory = Mock(return_value="# Memory\nA bank long enough")
        return curator

    def test_explicit_datetimes_reach_the_history_query_untouched(self):
        curator = self._curator()
        start, end = datetime(2026, 9, 10, 14, 3), datetime(2026, 9, 10, 16, 30)
        with patch('assistant.agents.LongTermMemoryCurator.CalendarAccess.current', return_value=_no_events()):
            _, metadata = curator.curate_memory(start, end)
        assert curator.floxy_db.get_interaction_history.call_args.args == (start, end)
        assert metadata['period_start'] == "2026-09-10T14:03"
        assert metadata['period_end'] == "2026-09-10T16:30"

    def test_period_is_told_to_the_llm(self):
        curator = self._curator(history="[2026-09-10 14:10:00] User: hi")
        with patch('assistant.agents.LongTermMemoryCurator.CalendarAccess.current', return_value=_no_events()):
            curator.curate_memory(datetime(2026, 9, 10, 14, 3), datetime(2026, 9, 10, 16, 30))
        prompt = curator.llm.think.call_args.kwargs['system_message']
        assert "This run covers 2026-09-10 14:03 to 2026-09-10 16:30." in prompt
        assert "{PERIOD}" not in prompt

    @pytest.mark.parametrize("voice_enabled", [True, False])
    def test_stt_note_only_when_the_voice_interface_is_enabled(self, voice_enabled):
        config = Mock()
        config.get.side_effect = lambda key, default=None: voice_enabled if key == 'interfaces.voice.enabled' else default
        curator = LongTermMemoryCurator(llm=Mock(), test_mode=True)
        curator.llm.think.return_value = "# Memory"
        with patch('assistant.agents.LongTermMemoryCurator.ConfigManager.get_instance', return_value=config):
            curator.curate_memory_with_data(interaction_history="", calendar_data="", current_memory="")
        prompt = curator.llm.think.call_args.kwargs['system_message']
        assert ("misheard phrases" in prompt) is voice_enabled
        assert "{STT_NOTE}" not in prompt

    def test_dates_mean_midnight(self):
        curator = self._curator()
        with patch('assistant.agents.LongTermMemoryCurator.CalendarAccess.current', return_value=_no_events()):
            curator.curate_memory(date(2026, 9, 8), date(2026, 9, 10))
        assert curator.floxy_db.get_interaction_history.call_args.args == (datetime(2026, 9, 8), datetime(2026, 9, 10))

    def test_default_period_is_lookback_midnight_until_now(self):
        curator = self._curator()
        with patch('assistant.agents.LongTermMemoryCurator.CalendarAccess.current', return_value=_no_events()):
            before = datetime.now()
            curator.curate_memory(lookback_days=2)
            after = datetime.now()
        start, end = curator.floxy_db.get_interaction_history.call_args.args
        assert start == datetime.combine(before.date() - timedelta(days=2), datetime.min.time())
        assert before <= end <= after

    def test_nightly_default_covers_yesterday_from_midnight(self):
        """The nightly run passes no period: it curates one day — yesterday from its midnight to now."""
        curator = self._curator()
        with patch('assistant.agents.LongTermMemoryCurator.CalendarAccess.current', return_value=_no_events()):
            before = datetime.now()
            curator.curate_memory()
            after = datetime.now()
        start, end = curator.floxy_db.get_interaction_history.call_args.args
        assert start == datetime.combine(before.date() - timedelta(days=1), datetime.min.time())
        assert before <= end <= after

    def test_empty_period_is_refused(self):
        curator = self._curator()
        with pytest.raises(ValueError):
            curator.curate_memory(datetime(2026, 9, 10, 16), datetime(2026, 9, 10, 14))

    def test_curate_with_data_has_no_period_placeholder_left(self):
        curator = self._curator()
        curator.curate_memory_with_data("history", "calendar", "# Memory\nA bank long enough")
        prompt = curator.llm.think.call_args.kwargs['system_message']
        assert "{PERIOD}" not in prompt
        assert "the recent interactions below" in prompt


class TestThePromptSaysWhatTheAiAlreadyHas:
    """The curator sees the bank and nothing else of the AI's prompt, so it is told what stands
    there live — or the bank repeats the integrations, the vault's labels and the like."""

    def _prompt(self, monitor) -> str:
        llm = Mock()
        llm.think.return_value = "# Memory\nA bank long enough to pass the size check for sure"
        curator = LongTermMemoryCurator(llm=llm, test_mode=True)
        with patch('assistant.integration.IntegrationMonitor.IntegrationMonitor.get_instance', return_value=monitor):
            curator.curate_memory_with_data("history", "", "# Memory\nA bank long enough")
        return llm.think.call_args_list[0].kwargs['system_message']

    def test_the_sections_and_toolkits_of_the_integrations_that_are_on_are_named(self):
        from assistant.integration.calendar.CalendarSource import CalendarSource
        from assistant.prompt.sections.CalendarSectionGenerator import CalendarSectionGenerator
        monitor = Mock()
        monitor.folder_enabled.side_effect = lambda folder: folder == "photos"
        monitor.contract_enabled.side_effect = lambda contract: contract is CalendarSource
        block = self._prompt(monitor).split("## Already in the AI's prompt")[1].split("## Your task")[0]
        assert CalendarSectionGenerator.heading.lstrip("# ") in block
        assert "CAR INFO" not in block, "no car integration is on"
        toolkits = next(line for line in block.splitlines() if line.startswith("- toolkits"))
        assert "photos" in toolkits and "calendar" in toolkits and "vault" in toolkits
        assert "dropbox" not in toolkits and "cars" not in toolkits

    def test_no_placeholder_of_the_block_is_left_when_nothing_is_on(self):
        monitor = Mock()
        monitor.folder_enabled.return_value = False
        monitor.contract_enabled.return_value = False
        prompt = self._prompt(monitor)
        assert "{IN_PROMPT}" not in prompt
        assert "- sections:" not in prompt, "no data section runs, so the line is left out"


class TestThePromptNamesNoSize:
    """The bank's size is kept by code (the condensing pass). Given a count and a limit the
    model tried to meet it by counting words and thought through its whole output cap
    (2026-11-13), so the curation prompt carries neither, whatever the size of the bank."""

    @pytest.mark.parametrize('words', [MEMORY_WORD_LIMIT - 899, MEMORY_WORD_LIMIT + 163,
                                       int(MEMORY_WORD_LIMIT * 1.3) + 7])
    def test_neither_the_limit_nor_the_measured_count_is_in_the_prompt(self, words):
        llm = Mock()
        llm.last_stop_reason = None
        llm.think.return_value = "<memory># Memory\nUpdated bank with enough words to pass the shrink check</memory>"
        curator = LongTermMemoryCurator(llm=llm, test_mode=True)
        memory = "# Memory\n" + " ".join(["word"] * (words - 2))
        assert curator.word_count(memory) == words
        curator.curate_memory_with_data("history", "", memory)
        prompt = curator.llm.think.call_args_list[0].kwargs['system_message']
        assert memory in prompt
        assert str(MEMORY_WORD_LIMIT) not in prompt and str(words) not in prompt
        assert str(abs(words - MEMORY_WORD_LIMIT)) not in prompt
        assert "{MEMORY_SIZE}" not in prompt and "{WORD_LIMIT}" not in prompt


def _bank(words: int) -> str:
    return "# Memory\n" + " ".join(f"w{i}" for i in range(words - 1))


class TestCondensingPass:
    """A curated bank well over the limit is condensed by a pass of its own after the curation
    (the MemoryCondenser, the bank alone, no interactions): the curation prompt only rewords an
    oversize bank, and a pass before it would leave the period's additions on top of the limit."""

    def _curator(self, replies: list[str]):
        llm = Mock()
        llm.last_stop_reason = None
        llm.think.side_effect = replies
        return LongTermMemoryCurator(llm=llm, test_mode=True)

    def test_a_pass_is_asked_for_more_than_the_excess_and_down_to_one_size(self):
        # The model delivers a part of its ask, so the ask aims under the limit — at the same
        # size whatever the bank holds
        small, large = MEMORY_WORD_LIMIT + 500, MEMORY_WORD_LIMIT * 2
        cut = LongTermMemoryCurator.condense_cut
        assert cut(small) > small - MEMORY_WORD_LIMIT
        assert small - cut(small) == large - cut(large)
        assert cut(10) == 0

    def test_trigger_is_above_the_limit(self):
        assert not LongTermMemoryCurator.needs_condensing(MEMORY_WORD_LIMIT + 1)
        assert LongTermMemoryCurator.needs_condensing(int(MEMORY_WORD_LIMIT * (1 + CONDENSE_TRIGGER)) + 1)

    def test_a_curated_bank_over_the_trigger_is_condensed_and_the_condensed_bank_returned(self):
        loaded = _bank(MEMORY_WORD_LIMIT - 100)
        curated = _bank(int(MEMORY_WORD_LIMIT * 1.3))
        condensed = _bank(MEMORY_WORD_LIMIT - 90)
        curator = self._curator([f"<memory>{curated}</memory>", f"<section>{condensed}</section>"])
        memory, metadata = curator.curate_memory_with_data("the period's history", "", loaded)
        first, second = [c.kwargs['system_message'] for c in curator.llm.think.call_args_list]
        assert "the period's history" in first and loaded in first
        assert "the period's history" not in second and curated in second      # one section: one condensing call
        assert memory.rstrip("\n") == condensed
        assert metadata['condense']['from'] == curator.word_count(curated)
        assert metadata['condense']['to'] == curator.word_count(condensed)
        assert metadata['old_memory'] == loaded  # the append carry-over compares with the bank as loaded
        assert metadata['new_memory_size'] == len(memory)

    def test_a_loaded_bank_over_the_trigger_is_curated_as_it_is(self):
        # The curation sees the whole bank and is asked for no cut; the cut comes after
        oversize = _bank(int(MEMORY_WORD_LIMIT * 1.3))
        curator = self._curator([f"<memory>{_bank(3000)}</memory>"])
        _, metadata = curator.curate_memory_with_data("history", "", oversize)
        assert curator.llm.think.call_count == 1
        prompt = curator.llm.think.call_args.kwargs['system_message']
        assert oversize in prompt
        assert metadata['condense'] is None

    def test_a_curated_bank_under_the_trigger_is_not_condensed(self):
        curator = self._curator([f"<memory>{_bank(MEMORY_WORD_LIMIT + 10)}</memory>"])
        _, metadata = curator.curate_memory_with_data("history", "", _bank(3000))
        assert curator.llm.think.call_count == 1
        assert metadata['condense'] is None

    def test_a_rejected_curation_is_not_condensed(self):
        oversize = _bank(int(MEMORY_WORD_LIMIT * 1.3))
        curator = self._curator(["<memory># Memory\ntoo short</memory>"])
        memory, metadata = curator.curate_memory_with_data("history", "", oversize)
        assert memory is oversize and curator.llm.think.call_count == 1
        assert metadata['condense'] is None

    def test_a_condensing_pass_that_removed_nothing_keeps_the_curated_bank(self):
        curated = _bank(int(MEMORY_WORD_LIMIT * 1.3))
        curator = self._curator([f"<memory>{curated}</memory>", "no section here"])
        memory, metadata = curator.curate_memory_with_data("history", "", _bank(3000))
        assert memory == curated and not metadata['truncated']
        assert metadata['condense']['to'] == metadata['condense']['from']

    def test_a_bank_still_over_the_trigger_gets_another_pass(self):
        curated, once, twice = _bank(MEMORY_WORD_LIMIT + 2000), _bank(MEMORY_WORD_LIMIT + 1000), _bank(MEMORY_WORD_LIMIT)
        curator = self._curator([f"<memory>{curated}</memory>", f"<section>{once}</section>",
                                 f"<section>{twice}</section>", "<section>never asked for</section>"])
        memory, metadata = curator.curate_memory_with_data("history", "", _bank(3000))
        assert memory.rstrip("\n") == twice and curator.llm.think.call_count == 3
        assert once in curator.llm.think.call_args_list[2].kwargs['system_message']   # the second pass reads the first's bank
        assert metadata['condense']['from'] == curator.word_count(curated)
        assert metadata['condense']['to'] == curator.word_count(twice)
        assert len(metadata['condense']['passes']) == 2

    def test_a_pass_that_delivers_next_to_nothing_ends_the_condensing_with_an_error(self):
        curated, barely = _bank(MEMORY_WORD_LIMIT + 2000), _bank(MEMORY_WORD_LIMIT + 1990)
        curator = self._curator([f"<memory>{curated}</memory>", f"<section>{barely}</section>",
                                 f"<section>{_bank(MEMORY_WORD_LIMIT)}</section>"])
        with patch("assistant.agents.LongTermMemoryCurator.error") as error:
            memory, _ = curator.curate_memory_with_data("history", "", _bank(3000))
        assert memory.rstrip("\n") == barely and curator.llm.think.call_count == 2
        error.assert_called_once()

    def test_the_passes_stop_at_their_limit_with_an_error_while_the_bank_is_still_over(self):
        # Every pass removes a good part of its ask and the bank is over the trigger all the same
        sizes = [MEMORY_WORD_LIMIT + 3000 - 700 * n for n in range(CONDENSE_PASSES + 2)]
        assert sizes[CONDENSE_PASSES] > MEMORY_WORD_LIMIT * (1 + CONDENSE_TRIGGER)
        curator = self._curator([f"<memory>{_bank(sizes[0])}</memory>"]
                                + [f"<section>{_bank(size)}</section>" for size in sizes[1:]])
        with patch("assistant.agents.LongTermMemoryCurator.error") as error:
            memory, metadata = curator.curate_memory_with_data("history", "", _bank(3000))
        assert curator.llm.think.call_count == 1 + CONDENSE_PASSES
        assert memory.rstrip("\n") == _bank(sizes[CONDENSE_PASSES])
        error.assert_called_once()

    def test_a_later_pass_that_comes_back_longer_keeps_the_bank_of_the_pass_before(self):
        curated, once, longer = (_bank(MEMORY_WORD_LIMIT + 2000), _bank(MEMORY_WORD_LIMIT + 1000),
                                 _bank(MEMORY_WORD_LIMIT + 1500))
        curator = self._curator([f"<memory>{curated}</memory>", f"<section>{once}</section>",
                                 f"<section>{longer}</section>", "<section>never asked for</section>"])
        with patch("assistant.agents.LongTermMemoryCurator.error") as error:
            memory, metadata = curator.curate_memory_with_data("history", "", _bank(3000))
        assert memory.rstrip("\n") == once, "the second pass added words: the first pass's bank is the result"
        assert curator.llm.think.call_count == 3, "a pass that removed nothing is not followed by another"
        assert metadata['condense']['to'] == curator.word_count(once)
        error.assert_called_once()

    def test_a_memory_log_that_cannot_be_written_does_not_cost_the_condensing(self, tmp_path, monkeypatch):
        import assistant.agents.LongTermMemoryCurator as curator_mod
        in_the_way = tmp_path / "log"
        in_the_way.write_text("a file where the folder should be")
        monkeypatch.setattr(curator_mod, "MEMORY_LOG_DIR", in_the_way)
        condensed = _bank(MEMORY_WORD_LIMIT - 90)
        curator = self._curator([f"<memory>{_bank(MEMORY_WORD_LIMIT + 2000)}</memory>", f"<section>{condensed}</section>"])
        memory, _ = curator.curate_memory_with_data("history", "", _bank(3000))
        assert memory.rstrip("\n") == condensed

    def test_the_bank_each_pass_got_is_kept_in_the_memory_log(self, tmp_path, monkeypatch):
        import assistant.agents.LongTermMemoryCurator as curator_mod
        monkeypatch.setattr(curator_mod, "MEMORY_LOG_DIR", tmp_path / "log")
        curated, once = _bank(MEMORY_WORD_LIMIT + 2000), _bank(MEMORY_WORD_LIMIT + 1000)
        curator = self._curator([f"<memory>{curated}</memory>", f"<section>{once}</section>",
                                 f"<section>{_bank(MEMORY_WORD_LIMIT)}</section>"])
        curator.curate_memory_with_data("history", "", _bank(3000))
        kept = sorted(path.read_text() for path in (tmp_path / "log").glob("*before_condense*"))
        assert kept == sorted([curated, once + "\n"])


class TestRunCondense:
    """The condensing pass on its own (`//memory_condense`): the live bank cut, saved with a
    journal and a history row, under the curation lock."""

    def _curator(self, tmp_path, reply: str, bank: str):
        path = tmp_path / "bank.md"
        path.write_text(bank)
        llm = Mock()
        llm.last_stop_reason = None
        llm.think.return_value = reply
        return LongTermMemoryCurator(llm=llm, test_mode=True, memory_file_path=str(path)), path

    def test_saves_the_condensed_bank_and_records_the_completion(self, tmp_path, monkeypatch):
        import assistant.agents.LongTermMemoryCurator as curator_mod
        monkeypatch.setattr(curator_mod, "MEMORY_LOG_DIR", tmp_path / "log")
        condensed = _bank(MEMORY_WORD_LIMIT - 500)
        curator, path = self._curator(tmp_path, f"<section>{condensed}</section>", _bank(MEMORY_WORD_LIMIT + 300))
        history = Mock()
        outcome = LongTermMemoryCurator.run_condense(history=history, curator=curator)
        assert outcome['to'] == curator.word_count(condensed) < outcome['from']
        assert path.read_text().rstrip("\n") == condensed
        journals = list((tmp_path / "log").glob("*_memory.md"))
        assert len(journals) == 1 and "Condensing pass" in journals[0].read_text()
        note = history.add_context_message.call_args.args[1]
        assert str(outcome['from']) in note and str(outcome['to']) in note

    def test_the_journal_lists_every_pass_with_the_sizes_it_went_between(self, tmp_path, monkeypatch):
        import assistant.agents.LongTermMemoryCurator as curator_mod
        monkeypatch.setattr(curator_mod, "MEMORY_LOG_DIR", tmp_path / "log")
        curator, _ = self._curator(tmp_path, None, _bank(MEMORY_WORD_LIMIT + 2017))
        curator.llm.think.side_effect = [f"<section>{_bank(MEMORY_WORD_LIMIT + 1009)}</section>",
                                         f"<section>{_bank(MEMORY_WORD_LIMIT - 13)}</section>"]
        outcome = LongTermMemoryCurator.run_condense(curator=curator)
        first, second = outcome['passes']
        assert first['to'] == second['from'] and (outcome['from'], outcome['to']) == (first['from'], second['to'])
        [journal] = (tmp_path / "log").glob("*_memory.md")
        lines = journal.read_text().splitlines()
        for one_pass in (first, second):
            assert any(str(one_pass['from']) in line and str(one_pass['to']) in line and str(one_pass['cut']) in line
                       for line in lines), f"the journal does not show the pass {one_pass['from']} → {one_pass['to']}"

    def test_the_given_cut_reaches_the_condenser(self, tmp_path):
        curator, _ = self._curator(tmp_path, f"<section>{_bank(100)}</section>", _bank(500))
        LongTermMemoryCurator.run_condense(cut=123, curator=curator)
        assert "123" in curator.llm.think.call_args.kwargs['user_message']

    def test_a_given_cut_is_one_pass_however_large_the_bank_stays(self, tmp_path):
        left = _bank(MEMORY_WORD_LIMIT + 1500)
        curator, path = self._curator(tmp_path, f"<section>{left}</section>", _bank(MEMORY_WORD_LIMIT + 2000))
        LongTermMemoryCurator.run_condense(cut=500, curator=curator)
        assert curator.llm.think.call_count == 1 and path.read_text().rstrip("\n") == left

    def test_a_pass_that_removed_nothing_leaves_the_bank_alone(self, tmp_path):
        bank = _bank(MEMORY_WORD_LIMIT + 300)
        curator, path = self._curator(tmp_path, "no section", bank)
        history = Mock()
        assert LongTermMemoryCurator.run_condense(history=history, curator=curator) is None
        assert path.read_text() == bank
        history.add_context_message.assert_not_called()

    def test_skipped_while_another_curation_holds_the_lock(self, tmp_path):
        curator, _ = self._curator(tmp_path, "<section>x</section>", _bank(500))
        LongTermMemoryCurator._curation_lock.acquire()
        try:
            assert LongTermMemoryCurator.run_condense(curator=curator) is None
        finally:
            LongTermMemoryCurator._curation_lock.release()
        curator.llm.think.assert_not_called()


def _mock_curator() -> Mock:
    """A curator stub whose command block runs nothing: `run_commands` answers (results, unsaved)."""
    curator = Mock()
    curator.run_commands.return_value = ([], [])
    return curator


class TestRun:
    def test_run_curates_saves_and_logs(self, tmp_path):
        curator = _mock_curator()
        curator.curate_memory.return_value = ("new bank", {'interaction_count': 3, 'new_memory_size': 8})
        metadata = LongTermMemoryCurator.run(datetime(2026, 9, 10, 14), datetime(2026, 9, 10, 16), curator=curator)
        curator.curate_memory.assert_called_once_with(datetime(2026, 9, 10, 14), datetime(2026, 9, 10, 16))
        curator.save_memory.assert_called_once_with("new bank", metadata)
        curator.save_debug_log.assert_called_once_with("new bank", metadata)

    def test_run_records_the_completion_in_the_history(self):
        curator = _mock_curator()
        curator.curate_memory.return_value = ("bank", {
            'interaction_count': 17, 'new_memory_size': 4,
            'period_start': '2026-09-08T00:00', 'period_end': '2026-09-10T04:00'})
        history = Mock()
        LongTermMemoryCurator.run(curator=curator, history=history)
        history.add_context_message.assert_called_once()
        role, content, _ = history.add_context_message.call_args.args
        assert role == "system" and history.add_context_message.call_args.kwargs['source'] == HISTORY_SOURCE
        assert content == ("Long term memory curation complete - memory updated "
                           "(17 interactions, 2026-09-08T00:00 .. 2026-09-10T04:00).")

    def test_second_run_while_one_is_in_progress_is_skipped(self):
        started, release = threading.Event(), threading.Event()
        slow = _mock_curator()

        def curate(*_):
            started.set()
            release.wait(FAILURE_BOUND)
            return "bank", {'interaction_count': 0, 'new_memory_size': 4}
        slow.curate_memory.side_effect = curate
        first = threading.Thread(target=LongTermMemoryCurator.run, kwargs={'curator': slow})
        first.start()
        assert started.wait(FAILURE_BOUND)
        try:
            assert LongTermMemoryCurator.run(curator=_mock_curator()) is None
        finally:
            release.set()
            first.join(FAILURE_BOUND)
        assert slow.save_memory.called

    def test_lock_is_released_after_a_failure(self):
        failing = _mock_curator()
        failing.curate_memory.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            LongTermMemoryCurator.run(curator=failing)
        ok = _mock_curator()
        ok.curate_memory.return_value = ("bank", {'interaction_count': 0, 'new_memory_size': 4})
        assert LongTermMemoryCurator.run(curator=ok) is not None


class TestSaveMemoryCarriesOverAppends:
    """A `remember` that lands while the LLM is thinking must survive the rewrite."""

    def test_appended_note_is_kept(self, tmp_path):
        memory_file = tmp_path / "memory.md"
        memory_file.write_text("# Bank\nold\n")
        curator = LongTermMemoryCurator(llm=Mock(), test_mode=True, memory_file_path=str(memory_file))
        loaded = curator._load_memory()
        MemoryFile.append("Marcus likes tea", str(memory_file))
        curator.save_memory("# Bank\ncurated\n", {'old_memory': loaded})
        assert memory_file.read_text() == "# Bank\ncurated\n\nMarcus likes tea\n"

    def test_unchanged_file_is_replaced_verbatim(self, tmp_path):
        memory_file = tmp_path / "memory.md"
        memory_file.write_text("# Bank\nold\n")
        curator = LongTermMemoryCurator(llm=Mock(), test_mode=True, memory_file_path=str(memory_file))
        curator.save_memory("# Bank\ncurated\n", {'old_memory': curator._load_memory()})
        assert memory_file.read_text() == "# Bank\ncurated\n"

    def test_rewritten_file_is_replaced_by_the_curated_bank(self, tmp_path):
        memory_file = tmp_path / "memory.md"
        memory_file.write_text("# Bank\nold\n")
        curator = LongTermMemoryCurator(llm=Mock(), test_mode=True, memory_file_path=str(memory_file))
        loaded = curator._load_memory()
        memory_file.write_text("# Something else entirely\n")
        curator.save_memory("# Bank\ncurated\n", {'old_memory': loaded})
        assert memory_file.read_text() == "# Bank\ncurated\n"

    def test_no_temp_file_is_left_behind(self, tmp_path):
        memory_file = tmp_path / "memory.md"
        curator = LongTermMemoryCurator(llm=Mock(), test_mode=True, memory_file_path=str(memory_file))
        curator.save_memory("# Bank\n")
        assert not [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


# --- chunked spans (run_span) --------------------------------------------------------------

class TestChunks:
    """`chunks` cuts a span of days into CURATION_CHUNK_DAYS-day periods, day-aligned with
    exclusive ends, the last one clamped to now."""

    NOW = datetime(2026, 9, 14, 11, 0)

    def _chunks(self, start, end, now=None):
        return LongTermMemoryCurator.chunks(start, end, now=now or self.NOW)

    def test_one_day_is_one_chunk(self):
        assert self._chunks(date(2026, 8, 1), date(2026, 8, 1)) == \
            [(datetime(2026, 8, 1), datetime(2026, 8, 2))]

    def test_exactly_four_days_is_one_chunk(self):
        assert self._chunks(date(2026, 8, 1), date(2026, 8, 4)) == \
            [(datetime(2026, 8, 1), datetime(2026, 8, 5))]

    def test_five_days_is_four_plus_one(self):
        assert self._chunks(date(2026, 8, 1), date(2026, 8, 5)) == [
            (datetime(2026, 8, 1), datetime(2026, 8, 5)),
            (datetime(2026, 8, 5), datetime(2026, 8, 6)),
        ]

    def test_nine_days_is_four_four_one(self):
        assert self._chunks(date(2026, 8, 1), date(2026, 8, 9)) == [
            (datetime(2026, 8, 1), datetime(2026, 8, 5)),
            (datetime(2026, 8, 5), datetime(2026, 8, 9)),
            (datetime(2026, 8, 9), datetime(2026, 8, 10)),
        ]

    def test_a_span_ending_today_stops_now(self):
        periods = self._chunks(date(2026, 9, 12), date(2026, 9, 14))
        assert periods == [(datetime(2026, 9, 12), self.NOW)]

    def test_an_end_in_the_future_is_clamped_to_today(self):
        periods = self._chunks(date(2026, 9, 13), date(2026, 9, 30))
        assert periods == [(datetime(2026, 9, 13), self.NOW)]

    def test_end_before_start_is_refused(self):
        with pytest.raises(ValueError):
            self._chunks(date(2026, 8, 5), date(2026, 8, 1))
        with pytest.raises(ValueError):   # start in the future, once the end is clamped
            self._chunks(date(2026, 9, 20), date(2026, 9, 25))

class TestLiveEnd:
    """`live_end` re-cuts a chunk end that `chunks` clamped to now at the now the chunk runs at."""

    def test_a_day_aligned_end_stays_whenever_it_runs(self):
        planned = datetime(2026, 8, 10)
        assert LongTermMemoryCurator.live_end(planned, now=datetime(2026, 8, 12, 9, 30)) == planned

    def test_an_end_cut_at_now_moves_to_the_now_it_runs_at(self):
        planned = datetime(2026, 10, 6, 18, 43)
        assert LongTermMemoryCurator.live_end(planned, now=datetime(2026, 10, 6, 18, 58)) == datetime(2026, 10, 6, 18, 58)

    def test_it_never_reaches_past_the_midnight_the_chunk_was_cut_from(self):
        planned = datetime(2026, 10, 6, 23, 50)
        assert LongTermMemoryCurator.live_end(planned, now=datetime(2026, 10, 7, 0, 20)) == datetime(2026, 10, 7)

    def test_it_never_moves_backwards(self):
        planned = datetime(2026, 10, 6, 18, 43)
        assert LongTermMemoryCurator.live_end(planned, now=datetime(2026, 10, 6, 18, 40)) == planned


def _span_curator(metadata_for=None):
    """A curator whose curate_memory answers ("bank", metadata) per call; `metadata_for(start)`
    picks the metadata by the chunk's start (default: a normal curated chunk)."""
    curator = _mock_curator()

    def curate(period_start, period_end):
        base = {'interaction_count': 2, 'new_memory_size': 8,
                'period_start': period_start.isoformat(timespec='minutes'),
                'period_end': period_end.isoformat(timespec='minutes')}
        extra = metadata_for(period_start) if metadata_for else {}
        if isinstance(extra, Exception):
            raise extra
        return "bank", {**base, **extra}
    curator.curate_memory.side_effect = curate
    return curator


class TestRunSpan:
    START, END = date(2026, 8, 1), date(2026, 8, 9)     # three chunks: 4 + 4 + 1 days

    def test_one_curation_per_chunk_in_order(self):
        curator = _span_curator()
        summary = LongTermMemoryCurator.run_span(self.START, self.END, curator=curator)
        assert [c.args for c in curator.curate_memory.call_args_list] == \
            LongTermMemoryCurator.chunks(self.START, self.END)
        assert curator.save_memory.call_count == 3 and curator.save_debug_log.call_count == 3
        assert summary == {'chunks': 3, 'curated': 3, 'empty': 0, 'failed': 0, 'aborted': False, 'interaction_count': 6,
                           'period_start': '2026-08-01T00:00', 'period_end': '2026-08-10T00:00'}

    def test_the_last_chunk_of_a_span_ending_today_ends_when_it_runs(self):
        curator = _span_curator()
        start, today = date.today() - timedelta(days=1), date.today()
        moved_end = datetime.now() + timedelta(minutes=15)
        with patch.object(LongTermMemoryCurator, "live_end", return_value=moved_end) as live:
            summary = LongTermMemoryCurator.run_span(start, today, curator=curator)
        assert live.call_count == 1 and live.call_args.args[0].date() == today   # the end chunks() cut at now
        assert curator.curate_memory.call_args.args[1] == moved_end
        assert summary['period_end'] == moved_end.isoformat(timespec='minutes')

    def test_empty_chunk_is_counted_and_not_saved(self):
        curator = _span_curator(lambda start: {'empty': True, 'interaction_count': 0}
                                if start == datetime(2026, 8, 5) else {})
        summary = LongTermMemoryCurator.run_span(self.START, self.END, curator=curator)
        assert summary['curated'] == 2 and summary['empty'] == 1 and summary['interaction_count'] == 4
        assert curator.save_memory.call_count == 2

    def test_failed_chunk_does_not_stop_the_span(self):
        curator = _span_curator(lambda start: RuntimeError("boom") if start == datetime(2026, 8, 1) else {})
        summary = LongTermMemoryCurator.run_span(self.START, self.END, curator=curator)
        assert summary['failed'] == 1 and summary['curated'] == 2
        assert curator.curate_memory.call_count == 3

    def test_truncated_chunk_counts_as_failed(self):
        curator = _span_curator(lambda start: {'truncated': True} if start == datetime(2026, 8, 9) else {})
        summary = LongTermMemoryCurator.run_span(self.START, self.END, curator=curator)
        assert summary['failed'] == 1 and summary['curated'] == 2
        assert curator.save_memory.call_count == 3   # a truncated reply saves the unchanged bank, as run does

    def test_on_chunk_is_called_one_based_with_none_on_failure(self):
        curator = _span_curator(lambda start: RuntimeError("boom") if start == datetime(2026, 8, 5) else {})
        seen = []
        LongTermMemoryCurator.run_span(self.START, self.END, curator=curator,
                                       on_chunk=lambda index, total, metadata: seen.append((index, total, metadata)))
        assert [(i, t) for i, t, _ in seen] == [(1, 3), (2, 3), (3, 3)]
        assert seen[1][2] is None
        assert seen[0][2]['period_start'] == '2026-08-01T00:00' and seen[2][2]['period_start'] == '2026-08-09T00:00'

    def test_first_chunk_resumes_a_span(self):
        curator = _span_curator()
        seen = []
        summary = LongTermMemoryCurator.run_span(self.START, self.END, curator=curator, first_chunk=2,
                                                 on_chunk=lambda i, t, m: seen.append((i, t)))
        assert curator.curate_memory.call_args_list == [call(datetime(2026, 8, 9), datetime(2026, 8, 10))]
        assert seen == [(3, 3)]
        assert summary['chunks'] == 3 and summary['curated'] == 1

    def test_on_chunk_returning_false_stops_the_span_without_a_row(self):
        curator = _span_curator()
        history = Mock()
        summary = LongTermMemoryCurator.run_span(self.START, self.END, curator=curator, history=history,
                                                 on_chunk=lambda i, t, m: i < 2)
        assert curator.curate_memory.call_count == 2
        assert summary['aborted'] is True and summary['curated'] == 2 and summary['chunks'] == 3
        history.add_context_message.assert_not_called()

    def test_on_chunk_returning_none_lets_the_span_run(self):
        curator = _span_curator()
        summary = LongTermMemoryCurator.run_span(self.START, self.END, curator=curator, on_chunk=lambda i, t, m: None)
        assert curator.curate_memory.call_count == 3 and summary['aborted'] is False

    def test_multi_chunk_span_records_one_span_row_and_no_chunk_rows(self):
        history = Mock()
        LongTermMemoryCurator.run_span(self.START, self.END, curator=_span_curator(), history=history)
        rows = [c.args[1] for c in history.add_context_message.call_args_list]
        assert rows == ["Long term memory curated over 1 August .. 10 August from 6 interactions."]
        assert history.add_context_message.call_args.kwargs['source'] == HISTORY_SOURCE

    def test_span_note_mentions_failed_chunks_only_when_there_were_any(self):
        summary = {'chunks': 3, 'curated': 1, 'empty': 0, 'failed': 2, 'interaction_count': 6,
                   'period_start': '2026-08-01T00:00', 'period_end': '2026-08-10T23:59'}
        assert LongTermMemoryCurator.span_note(summary) == ("Long term memory curated over 1 August .. 10 August from "
                                                            "6 interactions; 2 of its 3 chunks failed and left nothing.")

    def test_single_chunk_span_records_only_the_completion_row(self):
        history = Mock()
        LongTermMemoryCurator.run_span(date(2026, 8, 1), date(2026, 8, 3), curator=_span_curator(), history=history)
        assert history.add_context_message.call_count == 1

    def test_span_is_skipped_while_another_curation_runs(self):
        started, release = threading.Event(), threading.Event()
        slow = _mock_curator()

        def curate(*_):
            started.set()
            release.wait(FAILURE_BOUND)
            return "bank", {'interaction_count': 0, 'new_memory_size': 4}
        slow.curate_memory.side_effect = curate
        first = threading.Thread(target=LongTermMemoryCurator.run_span,
                                 args=(self.START, self.END), kwargs={'curator': slow})
        first.start()
        assert started.wait(FAILURE_BOUND)
        try:
            assert LongTermMemoryCurator.run_span(self.START, self.END, curator=_span_curator()) is None
            assert LongTermMemoryCurator.run(curator=_mock_curator()) is None
        finally:
            release.set()
            first.join(FAILURE_BOUND)
        assert slow.curate_memory.call_count == 3


class TestEmptyPeriod:
    """A period with no interactions costs no LLM call and changes nothing."""

    def _curator(self):
        llm = Mock()
        curator = LongTermMemoryCurator(llm=llm, test_mode=True)
        curator.floxy_db = Mock()
        curator.floxy_db.get_interaction_history.return_value = NO_INTERACTIONS
        curator._load_memory = Mock(return_value="# Memory\nA bank long enough")
        return curator

    def test_curate_memory_returns_the_bank_untouched(self):
        curator = self._curator()
        with patch('assistant.agents.LongTermMemoryCurator.CalendarAccess.current') as calendar:
            memory, metadata = curator.curate_memory(datetime(2026, 9, 10, 14), datetime(2026, 9, 10, 16))
        assert memory == "# Memory\nA bank long enough"
        assert metadata['empty'] is True and metadata['interaction_count'] == 0
        assert metadata['period_start'] == '2026-09-10T14:00' and metadata['period_end'] == '2026-09-10T16:00'
        assert metadata['old_memory_size'] == metadata['new_memory_size'] == len(memory)
        curator.llm.think.assert_not_called()
        calendar.assert_not_called()

    def test_run_does_not_save_or_record_an_empty_period(self):
        curator = Mock()
        curator.curate_memory.return_value = ("bank", {'empty': True, 'interaction_count': 0})
        history = Mock()
        metadata = LongTermMemoryCurator.run(curator=curator, history=history)
        assert metadata['empty'] is True
        curator.save_memory.assert_not_called()
        curator.save_debug_log.assert_not_called()
        history.add_context_message.assert_not_called()


class TestBatch:
    """The nightly run goes through the batch API; every other outcome is the ordinary call."""

    BANK = "# Memory\n" + "Marcus likes tea. " * 20

    def _curate(self, batch, batched_reply=None, batch_config=None, batched_error=None):
        llm = Mock()
        llm.last_stop_reason = 'end_turn'
        llm.think.return_value = self.BANK + "ordinary"
        llm.think_batched.return_value = batched_reply
        llm.think_batched.side_effect = batched_error
        curator = LongTermMemoryCurator(llm=llm, test_mode=True, batch=batch)
        values = {'interfaces.voice.enabled': False}
        if batch_config is not None:
            values.update(batch_config)
        config = Mock()
        config.get.side_effect = lambda key, default=None: values.get(key, default)
        with patch('assistant.agents.LongTermMemoryCurator.ConfigManager.get_instance', return_value=config), \
             patch('assistant.agents.LongTermMemoryCurator.WatchList.render', return_value=""), \
             patch('assistant.agents.LongTermMemoryCurator.Procedures.render_index', return_value=""):
            memory, _ = curator.curate_memory_with_data("[10:00] Marcus: hi", "", self.BANK)
        return memory, llm

    def test_batched_reply_is_the_bank_and_nothing_is_paid_twice(self):
        memory, llm = self._curate(batch=True, batched_reply=self.BANK + "batched")
        assert memory.endswith("batched")
        llm.think.assert_not_called()

    def test_no_answer_from_the_batch_falls_back_to_the_ordinary_call(self):
        memory, llm = self._curate(batch=True, batched_reply=None)
        assert memory.endswith("ordinary")

    def test_failed_batch_falls_back_to_the_ordinary_call(self):
        memory, _ = self._curate(batch=True, batched_error=RuntimeError("batches are down"))
        assert memory.endswith("ordinary")

    def test_refused_batch_is_not_sent_again(self):
        from assistant.llm.AbstractLLM import LLMRefusalError
        with pytest.raises(LLMRefusalError):
            self._curate(batch=True, batched_error=LLMRefusalError("refused"))

    def test_interactive_curation_never_waits_for_a_batch(self):
        _, llm = self._curate(batch=False)
        llm.think_batched.assert_not_called()

    def test_config_key_ai_memory_curator_batch_turns_it_off(self):
        """Contract with config.yaml: `ai.memory_curator.batch: false`."""
        memory, llm = self._curate(batch=True, batched_reply=self.BANK + "batched",
                                   batch_config={'ai.memory_curator.batch': False})
        assert memory.endswith("ordinary")
        llm.think_batched.assert_not_called()

    def test_config_key_batch_wait_minutes_is_the_wait_handed_to_the_llm(self):
        """Contract with config.yaml: `ai.memory_curator.batch_wait_minutes`."""
        _, llm = self._curate(batch=True, batched_reply=self.BANK,
                              batch_config={'ai.memory_curator.batch_wait_minutes': 20})
        assert llm.think_batched.call_args.kwargs['max_wait_seconds'] == 20 * 60

    def test_nightly_run_builds_a_batching_curator(self):
        with patch.object(LongTermMemoryCurator, '_curate_period', return_value={}) as curate, \
             patch('assistant.agents.LongTermMemoryCurator.create_llm'), \
             patch('assistant.agents.LongTermMemoryCurator.FloxyDatabase.get_instance'):
            LongTermMemoryCurator.run(batch=True)
        assert curate.call_args.args[0].batch is True


class TestBatchedCondensing:
    """The condensing after a nightly curation takes the batch API too, a batch a section —
    until a batch gives no answer: from then on the run makes ordinary calls."""

    CURATED = f"<memory>{_bank(int(MEMORY_WORD_LIMIT * 1.3))}</memory>"
    CONDENSED = f"<section>{_bank(3000)}</section>"

    def _curate(self, batched: list, ordinary: list = (), batch_config=None):
        llm = Mock()
        llm.last_stop_reason = None
        llm.think.side_effect = list(ordinary)
        llm.think_batched.side_effect = list(batched)
        curator = LongTermMemoryCurator(llm=llm, test_mode=True, batch=True)
        values = {'interfaces.voice.enabled': False, 'ai.memory_curator.batch': True, **(batch_config or {})}
        config = Mock()
        config.get.side_effect = lambda key, default=None: values.get(key, default)
        with patch('assistant.agents.LongTermMemoryCurator.ConfigManager.get_instance', return_value=config), \
             patch('assistant.agents.LongTermMemoryCurator.WatchList.render', return_value=""), \
             patch('assistant.agents.LongTermMemoryCurator.Procedures.render_index', return_value=""), \
             patch.object(LongTermMemoryCurator, '_keep_before_pass'):
            memory, _ = curator.curate_memory_with_data("history", "", _bank(3000))
        return memory, llm

    def test_the_sections_of_a_nightly_run_come_from_batches(self):
        memory, llm = self._curate(batched=[self.CURATED, self.CONDENSED])
        assert memory.rstrip("\n") == _bank(3000)
        llm.think.assert_not_called()

    def test_a_section_waits_far_shorter_for_its_batch_than_the_curation(self):
        _, llm = self._curate(batched=[self.CURATED, self.CONDENSED])
        curation, section = (c.kwargs['max_wait_seconds'] for c in llm.think_batched.call_args_list)
        assert section * 5 < curation

    def test_a_curation_whose_batch_gave_no_answer_is_condensed_by_ordinary_calls(self):
        memory, llm = self._curate(batched=[None], ordinary=[self.CURATED, self.CONDENSED])
        assert memory.rstrip("\n") == _bank(3000)
        assert llm.think_batched.call_count == 1

    def test_a_section_whose_batch_gave_no_answer_ends_the_batches_of_the_later_passes_too(self):
        first_pass = f"<section>{_bank(MEMORY_WORD_LIMIT + 600)}</section>"     # still over the trigger
        memory, llm = self._curate(batched=[self.CURATED, None], ordinary=[first_pass, self.CONDENSED])
        assert memory.rstrip("\n") == _bank(3000)
        assert llm.think_batched.call_count == 2 and llm.think.call_count == 2

    def test_config_key_ai_memory_curator_batch_turns_the_sections_off_as_well(self):
        """Contract with config.yaml: `ai.memory_curator.batch: false`."""
        _, llm = self._curate(batched=[], ordinary=[self.CURATED, self.CONDENSED],
                              batch_config={'ai.memory_curator.batch': False})
        llm.think_batched.assert_not_called()


class TestReferenceResultsLeaveTheTranscript:
    """The curator reads the pages and mails the assistant fetched, not the reference text it
    loaded on the way — the toolkit and procedure references, the preload notes in front of
    them, the log dumps and the network listings."""

    def test_the_curator_hands_the_filter_to_the_history_query(self):
        curator = LongTermMemoryCurator(llm=Mock(), test_mode=True)
        curator.floxy_db = Mock()
        curator.floxy_db.get_interaction_history.return_value = NO_INTERACTIONS
        with patch('assistant.agents.LongTermMemoryCurator.CalendarAccess.current', return_value=_no_events()):
            curator.curate_memory(datetime(2026, 9, 10, 14, 3), datetime(2026, 9, 10, 16, 30))
        assert curator.floxy_db.get_interaction_history.call_args.kwargs["skip_command_result"] is is_reference_result



@pytest.mark.unit
class TestCondensingPassFailure:
    def test_a_condensing_pass_that_raises_keeps_the_curated_bank(self):
        curated = _bank(int(MEMORY_WORD_LIMIT * 1.3))
        llm = Mock()
        llm.last_stop_reason = None
        llm.think.side_effect = [f"<memory>{curated}</memory>", RuntimeError("overloaded")]
        curator = LongTermMemoryCurator(llm=llm, test_mode=True)
        memory, metadata = curator.curate_memory_with_data("history", "", _bank(3000))
        assert memory.rstrip("\n") == curated
        assert metadata['new_memory_size'] == len(memory)


@pytest.mark.unit
class TestLastRunAt:
    """The dream journal's name is the one record of a run that survives a restart: the
    nightly trigger reads it back, so the writer and the reader must agree."""

    @pytest.fixture
    def journal_dir(self, tmp_path, monkeypatch):
        import assistant.agents.LongTermMemoryCurator as curator_mod
        monkeypatch.setattr(curator_mod, "MEMORY_LOG_DIR", tmp_path / "memory")
        return tmp_path / "memory"

    def test_the_newest_journal_wins_whatever_the_glob_order_and_odd_names_are_skipped(self, journal_dir):
        journal_dir.mkdir()
        for name in ("20260914_043100_ab_memory.md", "20260915_020000_cd_memory.md", "notes_memory.md",
                     "20260916_050000_ef_other.md"):
            (journal_dir / name).write_text("x")
        assert LongTermMemoryCurator.last_run_at() == datetime(2026, 9, 15, 2, 0)

    def test_no_journal_folder_yet_means_no_run_yet(self, journal_dir):
        assert LongTermMemoryCurator.last_run_at() is None

    def test_the_journal_save_debug_log_writes_is_the_run_last_run_at_reads_back(self, journal_dir):
        before = datetime.now().replace(microsecond=0)
        LongTermMemoryCurator.save_debug_log("# Memory\n", {
            'period_start': '', 'period_end': '', 'interaction_count': 0, 'calendar_events_count': 0,
            'old_memory_size': 0, 'new_memory_size': 9, 'llm_duration_seconds': 0.0, 'truncated': False,
            'condense': None, 'timestamp': 0.0, 'old_memory': '', 'full_prompt': ''})
        assert before <= LongTermMemoryCurator.last_run_at() <= datetime.now()
