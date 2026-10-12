"""The memory curator and the three-part memory: its reply is the bank plus a command block that
tidies the watch list and adds to the procedures, and `propose_split` / `apply_split` move an
old one-file bank into the three parts."""
from unittest.mock import Mock, patch

import pytest

import assistant.agents.LongTermMemoryCurator as curator_mod
from assistant import MemoryFile, Procedures, WatchList
from assistant.agents.LongTermMemoryCurator import LongTermMemoryCurator

OLD_BANK = ("# The Family\n* Marcus likes tea\n* Charlotte is allergic to cats\n"
            "* Parcel from shop.example, order 4711, arriving Thursday\n"
            "* Shop checkout: open the cart, click Pay, never press Back\n" + "* filler fact\n" * 20)


@pytest.fixture(autouse=True)
def no_live_memory(tmp_path, monkeypatch):
    """Nothing here may reach the repository's own memory/ or log/memory/."""
    monkeypatch.setattr(WatchList, "WATCHLIST_DIR", tmp_path / "default" / "watchlist")
    monkeypatch.setattr(WatchList, "LOG_DIR", tmp_path / "default" / "log" / "watchlist")
    monkeypatch.setattr(Procedures, "PROCEDURES_DIR", tmp_path / "default" / "procedures")
    monkeypatch.setattr(Procedures, "LOG_DIR", tmp_path / "default" / "log" / "procedures")
    monkeypatch.setattr(curator_mod, "MEMORY_DIR", tmp_path / "live")
    monkeypatch.setattr(curator_mod, "MEMORY_LOG_DIR", tmp_path / "memlog")


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "live" / "watchlist", tmp_path / "live" / "procedures"


def make_curator(tmp_path, dirs, replies=None, stop_reason=None) -> LongTermMemoryCurator:
    llm = Mock()
    llm.last_stop_reason = stop_reason
    if replies is not None:
        llm.think.side_effect = list(replies)
    bank = tmp_path / "live" / "bank.md"
    return LongTermMemoryCurator(llm=llm, test_mode=True, memory_file_path=str(bank),
                                 watchlist_dir=dirs[0], procedures_dir=dirs[1])


@pytest.mark.unit
class TestReplyParsing:
    def test_tagged_reply_splits_into_bank_and_commands(self):
        reply = '<memory>\n# Bank\n* Marcus likes tea\n</memory>\n<commands>\ndelete_watch("parcel", "done")\n</commands>'
        rest, commands = LongTermMemoryCurator._split_commands(reply)
        assert commands == 'delete_watch("parcel", "done")'
        assert LongTermMemoryCurator._extract_memory_content(rest) == "# Bank\n* Marcus likes tea"

    def test_a_plain_reply_is_all_bank_and_no_commands(self):
        rest, commands = LongTermMemoryCurator._split_commands("# Bank\n* Marcus likes tea")
        assert commands == ""
        assert LongTermMemoryCurator._extract_memory_content(rest) == "# Bank\n* Marcus likes tea"

    def test_an_unclosed_command_block_runs_to_the_end(self):
        rest, commands = LongTermMemoryCurator._split_commands('<memory>bank</memory>\n<commands>\nsave_watch("a", "b", "")')
        assert commands == 'save_watch("a", "b", "")'
        assert "save_watch(" not in rest

    def test_a_code_fence_inside_the_tagged_bank_stays_in_the_bank(self):
        reply = "<memory>\n# Bank\n```\nliteral block\n```\nafter\n</memory>"
        assert LongTermMemoryCurator._extract_memory_content(reply) == "# Bank\n```\nliteral block\n```\nafter"


@pytest.mark.unit
class TestRunCommands:
    def test_watch_and_procedure_commands_land_in_the_injected_directories(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs)
        WatchList.save("Dentist for Charlotte", "12:30", "2026-03-11", dirs[0], tmp_path / "wlog")
        results, _ = curator.run_commands(
            'save_watch("Parcel from shop.example", "order 4711, evening (16-22)", "2026-03-12")\n'
            'save_watch("dentist-for-charlotte", "moved to 14:00")\n'
            'save_procedure("shop-checkout", "paying at shop.example", """1) open the cart\n2) click "Pay")""")\n'
            'procedure_note("shop-checkout", "the push may need a retry")\n')
        assert len(results) == 4 and not any(r.startswith(("FAILED", "SKIPPED")) for r in results)
        items = {i.name: (i.note, i.due) for i in WatchList.items(dirs[0])}
        assert items == {"dentist-for-charlotte": ("moved to 14:00", "2026-03-11"),
                         "parcel-from-shop-example": ("order 4711, evening (16-22)", "2026-03-12")}
        text = Procedures.read("shop-checkout", dirs[1])
        assert '2) click "Pay")' in text and "the push may need a retry" in text

    def test_prose_and_markdown_dressing_around_commands_is_tolerated(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs)
        results, _ = curator.run_commands('Here is what I did:\n- `save_watch("Parcel", "ordered", "")`\nThat is all.')
        assert [i.name for i in WatchList.items(dirs[0])] == ["parcel"]
        assert WatchList.get("parcel", dirs[0]).due == ""
        assert sum(r.startswith("SKIPPED") for r in results) == 2

    def test_a_line_after_a_closed_command_is_not_glued_into_its_last_argument(self, tmp_path, dirs):
        """Regression: every non-command line after the first command was appended to it, so a
        forbidden `checked_watch` line or trailing prose ended up inside a due date or a note."""
        curator = make_curator(tmp_path, dirs)
        Procedures.save("shop-checkout", "paying", "the steps", dirs[1], tmp_path / "plog")
        results, _ = curator.run_commands(
            'save_watch("sofa", """Paid (order 12), shipped""", "2026-11-08")\n'
            'checked_watch("sofa")\n'
            'procedure_note("shop-checkout", "the push may need a retry")\n'
            'delete_procedure("shop-checkout")\n'
            'That is all.')
        sofa = WatchList.get("sofa", dirs[0])
        assert (sofa.note, sofa.due) == ("Paid (order 12), shipped", "2026-11-08")
        text = Procedures.read("shop-checkout", dirs[1])
        assert text.endswith("the push may need a retry\n") and "delete_procedure" not in text
        assert sum(r.startswith("SKIPPED") for r in results) == 3 and not any(r.startswith("FAILED") for r in results)

    def test_a_command_spread_over_several_lines_is_still_one_command(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs)
        results, _ = curator.run_commands(
            'save_watch("sofa (grey)",\n'
            '    """Paid, shipped\n'
            'tracking 1234""",\n'
            '    "2026-11-08")\n'
            'delete_watch("nothing", "x")')
        sofa = WatchList.get("sofa-grey", dirs[0])
        assert (sofa.note, sofa.due) == ("Paid, shipped\ntracking 1234", "2026-11-08")
        assert [r[:6] for r in results] == ["save_w", "FAILED"]

    def test_a_stray_quote_in_one_note_does_not_swallow_the_next_command(self, tmp_path, dirs):
        """Regression: a note holding an inch mark and a quoted phrase unbalanced the quotes,
        and every command after it was lost."""
        curator = make_curator(tmp_path, dirs)
        results, _ = curator.run_commands(
            'save_watch("tv-delivery", "TV 77" panel, order 1 (33 kg), press "Pay"", "2026-10-02")\n'
            'save_watch("sofa-delivery", """ordered, "paid" (link), arrives in November""")\n'
            'save_watch("parcel", "at the pickup point (code 12)")')
        items = {i.name: i for i in WatchList.items(dirs[0])}
        assert set(items) == {"tv-delivery", "sofa-delivery", "parcel"}
        assert items["tv-delivery"].note == 'TV 77" panel, order 1 (33 kg), press "Pay"'
        assert items["tv-delivery"].due == "2026-10-02"
        assert items["sofa-delivery"].note == 'ordered, "paid" (link), arrives in November'
        assert not any(r.startswith(("SKIPPED", "FAILED")) for r in results)

    def test_a_command_name_inside_a_procedure_text_does_not_start_a_command(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs)
        curator.run_commands('save_procedure("shop-checkout", "paying at shop.example", """1) open the cart\n'
                             'save_watch("x", "y") is what to emit afterwards\n2) click "Pay")""")')
        assert WatchList.items(dirs[0]) == []
        assert 'save_watch("x", "y") is what to emit afterwards' in Procedures.read("shop-checkout", dirs[1])

    def test_a_command_outside_the_allowlist_is_not_run(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs)
        Procedures.save("shop-checkout", "paying", "the steps", dirs[1], tmp_path / "plog")
        results, _ = curator.run_commands('delete_procedure("shop-checkout")\nremember("x")\nwatch_close(1, "y")')
        assert Procedures.read("shop-checkout", dirs[1]) is not None
        assert len(results) == 3
        assert all(r.startswith("SKIPPED") for r in results)

    def test_the_curator_cannot_stamp_an_item_as_checked(self, tmp_path, dirs):
        """Only a real check may move `checked`; the nightly rewrite never looked at anything."""
        curator = make_curator(tmp_path, dirs)
        WatchList.save("Tickets", "not on sale yet", None, dirs[0], tmp_path / "wlog", every="1d")
        path = dirs[0] / "tickets.md"
        path.write_text(path.read_text().replace(WatchList.get("Tickets", dirs[0]).checked, "2026-03-01 09:00"))
        results, _ = curator.run_commands('checked_watch("tickets")\nsave_watch("tickets", """on sale Friday""")')
        assert results[0] == 'SKIPPED (not a curator command): checked_watch("tickets")'
        tickets = WatchList.get("Tickets", dirs[0])
        assert tickets.note == "on sale Friday"
        assert tickets.checked == "2026-03-01 09:00"   # the rewrite is not a look; the assistant's save_watch is
        WatchList.run("save_watch", ["tickets", "bought"], dirs[0], tmp_path / "wlog")
        assert WatchList.get("Tickets", dirs[0]).checked != "2026-03-01 09:00"

    def test_replacing_an_existing_procedure_is_refused_and_reported(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs)
        Procedures.save("shop-checkout", "paying", "the original steps", dirs[1], tmp_path / "plog")
        results, unsaved = curator.run_commands('save_procedure("Shop Checkout", "paying", """condensed""")')
        assert "the original steps" in Procedures.read("shop-checkout", dirs[1])
        assert len(results) == 1 and results[0].startswith("FAILED")
        assert len(unsaved) == 1 and "condensed" in unsaved[0] and "Shop Checkout" in unsaved[0]

    def test_a_refused_procedure_with_no_text_leaves_nothing_to_keep(self, tmp_path, dirs):
        _, unsaved = make_curator(tmp_path, dirs).run_commands('save_procedure("x", "when", "")')
        assert unsaved == []

    def test_a_failing_command_does_not_stop_the_ones_after_it(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs)
        results, _ = curator.run_commands('delete_watch("plumber", "gone")\nsave_watch("Parcel", "ordered", "")')
        assert results[0].startswith("FAILED")
        assert [i.name for i in WatchList.items(dirs[0])] == ["parcel"]

    def test_an_empty_block_does_nothing(self, tmp_path, dirs):
        assert make_curator(tmp_path, dirs).run_commands("  \n") == ([], [])
        assert not dirs[0].exists()


@pytest.mark.unit
class TestCuration:
    def test_the_prompt_shows_the_open_watch_items_and_the_procedure_index(self, tmp_path, dirs):
        WatchList.save("Parcel from shop.example", "order 4711", "2026-03-12", dirs[0], tmp_path / "wlog")
        Procedures.save("shop-checkout", "paying at shop.example", "SECRET STEP BODY", dirs[1], tmp_path / "plog")
        curator = make_curator(tmp_path, dirs, replies=["<memory># Bank</memory>"])
        curator.curate_memory_with_data("history", "", "# Bank")
        prompt = curator.llm.think.call_args.kwargs['system_message']
        assert "parcel-from-shop-example" in prompt and "order 4711" in prompt
        assert "shop-checkout" in prompt and "paying at shop.example" in prompt
        assert "SECRET STEP BODY" not in prompt
        for placeholder in ("{WATCHLIST}", "{PROCEDURES}", "{TODAY}"):
            assert placeholder not in prompt

    def test_the_command_block_travels_in_the_metadata_not_in_the_bank(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs, replies=[
            '<memory>\n# Bank\n* Marcus likes tea\n</memory>\n<commands>\nsave_watch("Parcel", "ordered", "")\n</commands>'])
        bank, metadata = curator.curate_memory_with_data("history", "", "# Bank\n* Marcus likes tea")
        assert bank == "# Bank\n* Marcus likes tea"
        assert metadata['commands'] == 'save_watch("Parcel", "ordered", "")'
        assert WatchList.items(dirs[0]) == []   # curating alone runs nothing

    def test_a_truncated_reply_carries_no_commands(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs, stop_reason='max_tokens', replies=[
            '<memory>\n' + "A" * 900 + '\n</memory>\n<commands>\nsave_watch("Parcel", "ordered", "")'])
        bank, metadata = curator.curate_memory_with_data("history", "", "B" * 1000)
        assert bank == "B" * 1000
        assert metadata['commands'] == ""

    def test_a_suspiciously_short_bank_carries_no_commands_either(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs, replies=['<memory>x</memory><commands>delete_watch("parcel", "y")</commands>'])
        _, metadata = curator.curate_memory_with_data("history", "", "B" * 1000)
        assert metadata['commands'] == ""

    def test_a_period_run_runs_the_commands_and_saves_the_bank(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs)
        WatchList.save("Parcel", "shipped", None, dirs[0], tmp_path / "wlog")
        curator.curate_memory = Mock(return_value=("# Bank\n", {
            'interaction_count': 2, 'new_memory_size': 7, 'commands': 'delete_watch("parcel", "delivered")'}))
        curator.save_debug_log = Mock()
        metadata = LongTermMemoryCurator.run(curator=curator)
        assert MemoryFile.read(curator.memory_file_path) == "# Bank\n"
        assert WatchList.items(dirs[0]) == []
        assert len(metadata['command_results']) == 1
        curator.save_debug_log.assert_called_once()

    def test_a_refused_procedure_keeps_its_text_in_the_bank(self, tmp_path, dirs):
        """Regression: the bank (now holding only a pointer to the how-to) was written before
        the commands ran, so a refused save_procedure lost the how-to from live memory."""
        curator = make_curator(tmp_path, dirs)
        Procedures.save("shop-checkout", "paying", "the original steps", dirs[1], tmp_path / "plog")
        curator.curate_memory = Mock(return_value=("# Bank\n* Shopping: see shop-checkout\n", {
            'interaction_count': 2, 'new_memory_size': 7,
            'commands': 'save_procedure("shop-checkout", "paying", """1) cart\n2) Pay, never Back""")'}))
        curator.save_debug_log = Mock()
        metadata = LongTermMemoryCurator.run(curator=curator)
        bank = MemoryFile.read(curator.memory_file_path)
        assert bank.startswith("# Bank\n* Shopping: see shop-checkout\n")
        assert "1) cart\n2) Pay, never Back" in bank and "shop-checkout" in bank.split("\n\n", 1)[1]
        assert metadata['new_memory_size'] == len(bank)
        assert "the original steps" in Procedures.read("shop-checkout", dirs[1])
        assert metadata['command_results'][0].startswith("FAILED")

    def test_an_empty_period_runs_no_commands(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs)
        WatchList.save("Parcel", "shipped", None, dirs[0], tmp_path / "wlog")
        curator.curate_memory = Mock(return_value=("# Bank\n", {'empty': True, 'commands': 'delete_watch("parcel", "x")'}))
        LongTermMemoryCurator.run(curator=curator)
        assert len(WatchList.items(dirs[0])) == 1

    def test_the_debug_log_lists_the_commands_that_ran(self, tmp_path, dirs, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = LongTermMemoryCurator.save_debug_log("# Bank\n", {
            'command_results': ['delete_watch(parcel, delivered)', 'FAILED delete_watch(plumber, x) — no such item']})
        text = open(path).read()
        assert "delete_watch(parcel, delivered)" in text and "FAILED delete_watch(plumber, x)" in text


EXTRACT_REPLY = ('<commands>\n'
                 'save_watch("Parcel from shop.example", "order 4711, arriving Thursday", "2026-03-12")\n'
                 'save_procedure("shop-checkout", "paying at shop.example", '
                 '"""open the cart, click Pay, never press Back""")\n'
                 '</commands>')
BANK_REPLY = "<memory>\n# The Family\n* Marcus likes tea\n* Charlotte is allergic to cats\n</memory>"


@pytest.mark.unit
class TestProposeSplit:
    def test_the_proposal_is_laid_out_like_the_memory_directory(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs, replies=[EXTRACT_REPLY, BANK_REPLY])
        MemoryFile.write(OLD_BANK, curator.memory_file_path)
        target = tmp_path / "proposal"
        summary = curator.propose_split(target)
        assert (target / "bank.md").read_text().strip() == "# The Family\n* Marcus likes tea\n* Charlotte is allergic to cats"
        assert [i.name for i in WatchList.items(target / "watchlist")] == ["parcel-from-shop-example"]
        assert "never press Back" in Procedures.read("shop-checkout", target / "procedures")
        assert summary['watch_items'] == 1 and summary['procedures'] == 1 and summary['failed_commands'] == 0

    def test_nothing_live_is_touched(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs, replies=[EXTRACT_REPLY, BANK_REPLY])
        MemoryFile.write(OLD_BANK, curator.memory_file_path)
        curator.propose_split(tmp_path / "proposal")
        assert MemoryFile.read(curator.memory_file_path) == OLD_BANK
        assert WatchList.items(dirs[0]) == [] and Procedures.names(dirs[1]) == []

    def test_the_second_pass_is_shown_what_the_first_lifted_out(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs, replies=[EXTRACT_REPLY, BANK_REPLY])
        MemoryFile.write(OLD_BANK, curator.memory_file_path)
        curator.propose_split(tmp_path / "proposal")
        first, second = (c.kwargs['system_message'] for c in curator.llm.think.call_args_list)
        assert OLD_BANK.strip() in first and OLD_BANK.strip() in second
        assert "order 4711, arriving Thursday" in second.split(OLD_BANK.strip(), 1)[1]
        assert "shop-checkout" in second.split(OLD_BANK.strip(), 1)[1]

    def test_a_truncated_reply_aborts_the_proposal(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs, stop_reason='max_tokens', replies=[EXTRACT_REPLY, BANK_REPLY])
        MemoryFile.write(OLD_BANK, curator.memory_file_path)
        with pytest.raises(RuntimeError):
            curator.propose_split(tmp_path / "proposal")
        assert not (tmp_path / "proposal" / "bank.md").exists()

    def test_no_bank_means_nothing_to_split(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs, replies=[EXTRACT_REPLY, BANK_REPLY])
        with pytest.raises(RuntimeError):
            curator.propose_split(tmp_path / "proposal")
        curator.llm.think.assert_not_called()


@pytest.mark.unit
class TestApplySplit:
    @pytest.fixture
    def proposal(self, tmp_path):
        source = tmp_path / "proposal"
        source.mkdir()
        (source / "bank.md").write_text("# The Family\n* Marcus likes tea\n")
        WatchList.save("Parcel from shop.example", "order 4711", "2026-03-12", source / "watchlist", tmp_path / "wlog")
        WatchList.save("Dentist for Charlotte", "12:30", None, source / "watchlist", tmp_path / "wlog")
        Procedures.save("shop-checkout", "paying", "proposed steps", source / "procedures", tmp_path / "plog")
        Procedures.save("school-portal-login", "school", "login steps", source / "procedures", tmp_path / "plog")
        return source

    def test_the_bank_is_replaced_and_the_items_and_procedures_put_in_place(self, tmp_path, dirs, proposal):
        curator = make_curator(tmp_path, dirs)
        MemoryFile.write(OLD_BANK, curator.memory_file_path)
        done = curator.apply_split(proposal)
        assert MemoryFile.read(curator.memory_file_path) == "# The Family\n* Marcus likes tea\n"
        assert {i.name: (i.note, i.due) for i in WatchList.items(dirs[0])} == {
            "parcel-from-shop-example": ("order 4711", "2026-03-12"), "dentist-for-charlotte": ("12:30", "")}
        assert Procedures.names(dirs[1]) == ["school-portal-login", "shop-checkout"]
        assert sorted(done['watch_items']) == ["dentist-for-charlotte", "parcel-from-shop-example"]
        assert sorted(done['procedures']) == ["school-portal-login", "shop-checkout"]
        assert done['kept'] == []

    def test_the_memory_as_it_was_is_backed_up_first(self, tmp_path, dirs, proposal):
        curator = make_curator(tmp_path, dirs)
        MemoryFile.write(OLD_BANK, curator.memory_file_path)
        done = curator.apply_split(proposal)
        backups = list((tmp_path / "memlog").glob("before_split_*"))
        assert len(backups) == 1 and str(backups[0]) == done['backup']
        assert (backups[0] / "bank.md").read_text() == OLD_BANK

    def test_an_open_watch_item_is_kept_as_it_is(self, tmp_path, dirs, proposal):
        curator = make_curator(tmp_path, dirs)
        WatchList.save("Dentist for Charlotte", "the live note: moved to 14:00", None, dirs[0], tmp_path / "wlog")
        WatchList.save("Plumber", "coming Friday", None, dirs[0], tmp_path / "wlog")
        done = curator.apply_split(proposal)
        notes = {i.name: i.note for i in WatchList.items(dirs[0])}
        assert notes == {"dentist-for-charlotte": "the live note: moved to 14:00", "plumber": "coming Friday",
                         "parcel-from-shop-example": "order 4711"}
        assert done['watch_items'] == ["parcel-from-shop-example"] and done['kept'] == ["dentist-for-charlotte"]

    def test_a_hand_edited_watch_item_in_the_proposal_is_copied_as_edited(self, tmp_path, dirs, proposal):
        edited = proposal / "watchlist" / "parcel-from-shop-example.md"
        edited.write_text(edited.read_text() + "reference restored by hand\n")
        make_curator(tmp_path, dirs).apply_split(proposal)
        assert "reference restored by hand" in WatchList.get("parcel-from-shop-example", dirs[0]).note

    def test_an_existing_procedure_is_kept_as_it_is(self, tmp_path, dirs, proposal):
        curator = make_curator(tmp_path, dirs)
        Procedures.save("shop-checkout", "paying", "the live steps", dirs[1], tmp_path / "plog")
        done = curator.apply_split(proposal)
        assert "the live steps" in Procedures.read("shop-checkout", dirs[1])
        assert done['kept'] == ["shop-checkout"] and done['procedures'] == ["school-portal-login"]

    def test_a_hand_edited_procedure_in_the_proposal_is_copied_as_edited(self, tmp_path, dirs, proposal):
        edited = proposal / "procedures" / "shop-checkout.md"
        edited.write_text(edited.read_text() + "\nstep restored by hand\n")
        curator = make_curator(tmp_path, dirs)
        curator.apply_split(proposal)
        assert (dirs[1] / "shop-checkout.md").read_text() == edited.read_text()

    def test_a_directory_without_a_bank_is_not_a_proposal(self, tmp_path, dirs):
        curator = make_curator(tmp_path, dirs)
        MemoryFile.write(OLD_BANK, curator.memory_file_path)
        with pytest.raises(RuntimeError):
            curator.apply_split(tmp_path / "nothing")
        assert MemoryFile.read(curator.memory_file_path) == OLD_BANK


def _config(tips: str) -> Mock:
    """A config whose only setting is the tips destination (owner Marcus, as the conftest pins)."""
    cfg = Mock()
    cfg.get.side_effect = lambda key, default=None: tips if key == curator_mod.TIPS_KEY else default
    cfg.get_owner_name.return_value = "Marcus"
    return cfg


@pytest.fixture
def board():
    """A registered group destination that records what it is sent; the registry is emptied after."""
    from assistant.Destinations import Destination, KIND_GROUP
    sent = []
    curator_mod.Destinations.register(Destination("Marcus & Alice board", KIND_GROUP,
                                                  send=lambda text: sent.append(text) or True))
    yield sent
    curator_mod.Destinations.clear()


@pytest.mark.unit
class TestTips:
    """The one message the curator may send: a tip for other households' Keeps, to the configured group."""

    def test_a_tip_goes_to_the_configured_group_vault_redacted(self, tmp_path, dirs, board, monkeypatch):
        monkeypatch.setattr(curator_mod, "redact_for_storage", lambda text: text.replace("4111 1111", "[card]"))
        curator = make_curator(tmp_path, dirs)
        with patch.object(curator_mod.ConfigManager, "get_instance", return_value=_config("Marcus & Alice board")):
            results, _ = curator.run_commands(
                'post_message("Marcus & Alice board", """Tip from Marcus\'s Keep: shop.example takes card 4111 1111, '
                'skip the guest checkout""")')
        assert board == ["Tip from Marcus's Keep: shop.example takes card [card], skip the guest checkout"]
        assert len(results) == 1 and results[0].startswith("post_message(")

    def test_the_destination_the_model_names_is_overridden_by_the_configured_one(self, tmp_path, dirs, board):
        curator = make_curator(tmp_path, dirs)
        with patch.object(curator_mod.ConfigManager, "get_instance", return_value=_config("Marcus & Alice board")):
            results, _ = curator.run_commands('post_message("owner", "Tip from Marcus\'s Keep: a tip")')
        assert board == ["Tip from Marcus's Keep: a tip"]
        assert not results[0].startswith("FAILED")

    def test_without_a_destination_a_post_is_skipped_and_nothing_is_sent(self, tmp_path, dirs, board):
        curator = make_curator(tmp_path, dirs)
        with patch.object(curator_mod.ConfigManager, "get_instance", return_value=_config("")):
            results, _ = curator.run_commands('post_message("Marcus & Alice board", "a tip")\ndelete_watch("x", "y")')
        assert board == []
        assert results[0].startswith("SKIPPED (tips are off)")
        assert len(results) == 2   # the command after it still runs

    def test_an_unknown_configured_group_is_a_failure_in_the_run_log(self, tmp_path, dirs, board):
        curator = make_curator(tmp_path, dirs)
        with patch.object(curator_mod.ConfigManager, "get_instance", return_value=_config("Nowhere")):
            results, _ = curator.run_commands('post_message("Nowhere", "a tip")')
        assert board == []
        assert results[0].startswith("FAILED") and "unknown destination" in results[0]

    @pytest.mark.parametrize("tips", ["Marcus & Alice board", ""])
    def test_the_prompt_asks_for_tips_only_when_a_destination_is_set(self, tmp_path, dirs, tips):
        curator = make_curator(tmp_path, dirs, replies=["# Memory"])
        with patch.object(curator_mod.ConfigManager, "get_instance", return_value=_config(tips)):
            curator.curate_memory_with_data(interaction_history="", calendar_data="", current_memory="")
        prompt = curator.llm.think.call_args.kwargs['system_message']
        assert "{TIPS}" not in prompt
        assert ("Tips for other Keeps" in prompt) is bool(tips)
        if tips:
            assert 'post_message("Marcus & Alice board"' in prompt
            assert "Tip from Marcus's Keep" in prompt
